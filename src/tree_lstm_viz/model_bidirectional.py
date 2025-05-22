import torch
import torch.nn as nn
import spacy
from typing import List, Dict, Any, Optional, Tuple

# Import our utilities
try:
    from src.tree_lstm_viz.logger import app_logger
    from src.tree_lstm_viz.benepar_utils import BeneparHelper
    structured_logging = True
except ImportError:
    import warnings
    structured_logging = False
    app_logger = None
    warnings.warn("Logger or BeneparHelper modules not found.")

# Force CPU mode to avoid CUDA compatibility issues
torch.cuda.is_available = lambda: False

class Node:
    """Tree node for constituency parsing tree."""
    def __init__(self, label: str, span: tuple, children: List['Node'] = None):
        self.label = label
        self.span = span  # (start_idx, end_idx) in spaCy tokens
        self.children = children or []
        # Bottom-up states
        self.h_bottom_up = None 
        self.c_bottom_up = None
        # Top-down states
        self.h_top_down = None
        self.c_top_down = None
        # Combined state
        self.h = None
        self.c = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return not self.children
    
    def __repr__(self) -> str:
        return f"{self.label} {self.span}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        return {
            'label': self.label,
            'span': self.span,
            'h': self.h.tolist() if self.h is not None else None,
            'c': self.c.tolist() if self.c is not None else None,
            'h_bottom_up': self.h_bottom_up.tolist() if self.h_bottom_up is not None else None,
            'h_top_down': self.h_top_down.tolist() if self.h_top_down is not None else None,
            'children': [child.to_dict() for child in self.children]
        }

def build_tree(span) -> Node:
    """
    Recursively build Node tree from a benepar-annotated spaCy Span.
    Span should have benepar extensions: span._.labels, span._.children, span.start, span.end
    """
    # If no sub-constituents, it's a leaf token
    children_spans = list(span._.children)
    if not children_spans:
        return Node("TOKEN", (span.start, span.end))

    # Otherwise, build child Nodes
    children = [build_tree(child) for child in children_spans]
    # Use the first label on this span (most trees have exactly one)
    label = span._.labels[0] if span._.labels else "UNKNOWN"
    return Node(label, (span.start, span.end), children)

class ChildSumTreeLSTM(nn.Module):
    """
    Child-Sum Tree-LSTM module that composes representations bottom-up.
    
    This implementation follows the Child-Sum Tree-LSTM architecture from
    "Improved Semantic Representations From Tree-Structured LSTM Networks"
    (Tai et al., 2015).
    """
    def __init__(self, d_in=768, d_hidden=768):
        super().__init__()
        self.d_hidden = d_hidden
        
        # Input, output, update gates (i, o, u)
        self.W_iou = nn.Linear(d_in, 3 * d_hidden)
        self.U_iou = nn.Linear(d_hidden, 3 * d_hidden, bias=False)
        
        # Forget gate (f) - separate for each child
        self.W_f = nn.Linear(d_in, d_hidden)
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)
    
    def forward(self, node: Node, vec_lookup: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recursively process the tree starting from the given node in a bottom-up manner.
        
        Args:
            node: A Node object with .span, .children attributes
            vec_lookup: Dictionary mapping token indices to their embeddings
            
        Returns:
            Tuple of (h, c) tensors representing the hidden and cell states
        """
        # Leaf case: inject token embedding and initialize cell state
        if node.is_leaf():
            # Get embedding for this token index
            token_idx = node.span[0]
            if token_idx not in vec_lookup:
                # Fallback for missing embeddings
                h = torch.zeros(self.d_hidden, device=next(self.parameters()).device)
            else:
                h = vec_lookup[token_idx]
            
            # Initialize cell state to zeros
            c = torch.zeros_like(h)
            node.h_bottom_up, node.c_bottom_up = h, c
            return h, c
        
        # Internal node: compose children recursively
        child_states = [self.forward(child, vec_lookup) for child in node.children]
        
        # Separate hidden and cell states
        h_children = torch.stack([h for h, _ in child_states])
        c_children = torch.stack([c for _, c in child_states])
        
        # Sum of child hidden states
        h_sum = h_children.sum(dim=0)
        
        # Average input representation
        x_j = h_sum / len(child_states)
        
        # Compute input, output, update gates
        iou = self.W_iou(x_j) + self.U_iou(h_sum)
        i, o, u = iou.chunk(3, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        
        # Initialize cell state with input modulation
        c = i * u
        
        # Add forget-gated child contributions
        for idx, (child, (h_k, c_k)) in enumerate(zip(node.children, child_states)):
            f_k = torch.sigmoid(self.W_f(x_j) + self.U_f(h_k))
            c = c + f_k * c_k
        
        # Compute hidden state from cell
        h = o * torch.tanh(c)
        
        # Store states in node
        node.h_bottom_up, node.c_bottom_up = h, c
        return h, c

class TopDownTreeLSTM(nn.Module):
    """
    Top-Down Tree-LSTM module that composes representations from parent to children.
    
    This implementation processes information from the root node down to the leaves,
    allowing information to flow from parent to children.
    """
    def __init__(self, d_in=768, d_hidden=768):
        super().__init__()
        self.d_hidden = d_hidden
        
        # Input, output, update gates (i, o, u)
        self.W_iou = nn.Linear(d_in, 3 * d_hidden)
        self.U_iou = nn.Linear(d_hidden, 3 * d_hidden, bias=False)
        
        # Forget gate (f) - for parent information
        self.W_f = nn.Linear(d_in, d_hidden)
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)
    
    def forward(self, node: Node, vec_lookup: Dict[int, torch.Tensor], 
                parent_h: Optional[torch.Tensor] = None, 
                parent_c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the tree top-down starting from the root node.
        
        Args:
            node: Current node being processed
            vec_lookup: Dictionary mapping token indices to embeddings
            parent_h: Hidden state from parent node (None for root)
            parent_c: Cell state from parent node (None for root)
            
        Returns:
            Tuple of (h, c) tensors representing the hidden and cell states
        """
        # Get input embedding for current node
        token_idx = node.span[0]
        if token_idx not in vec_lookup:
            # Fallback for missing embeddings
            x = torch.zeros(self.d_hidden, device=next(self.parameters()).device)
        else:
            x = vec_lookup[token_idx]
        
        # Root node case (no parent)
        if parent_h is None or parent_c is None:
            # For root node, compute gates using only the node's embedding
            iou = self.W_iou(x)
            i, o, u = iou.chunk(3, dim=-1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            
            # Initialize cell state
            c = i * u
            
            # Compute hidden state
            h = o * torch.tanh(c)
        else:
            # Compute input, output, update gates using both node embedding and parent hidden state
            iou = self.W_iou(x) + self.U_iou(parent_h)
            i, o, u = iou.chunk(3, dim=-1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            
            # Compute forget gate for parent information
            f = torch.sigmoid(self.W_f(x) + self.U_f(parent_h))
            
            # Compute cell state - combine new info with selective parent info
            c = i * u + f * parent_c
            
            # Compute hidden state
            h = o * torch.tanh(c)
        
        # Store states in node
        node.h_top_down, node.c_top_down = h, c
        
        # Process all children recursively with current node as parent
        for child in node.children:
            self.forward(child, vec_lookup, h, c)
        
        return h, c

class BidirectionalTreeLSTMEncoder:
    """Encode sentences using both bottom-up and top-down Tree-LSTM over constituency parse trees."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', mode: str = 'both'):
        """
        Initialize the bidirectional Tree-LSTM encoder.
        
        Args:
            model_name: Name of the BERT model to use as embedding provider
            mode: 'bottom-up', 'top-down', or 'both' to select which Tree-LSTM direction(s) to use
        """
        from transformers import BertModel, BertTokenizerFast
        
        # Set device (CPU only for compatibility)
        self.device = torch.device('cpu')
        self.mode = mode
        
        if structured_logging:
            app_logger.info(f"Running on: {self.device}")
            app_logger.info(f"Using Tree-LSTM mode: {mode}")
        
        # Initialize Benepar helper
        if structured_logging:
            self.benepar_helper = BeneparHelper('benepar_en3')
            # Ensure Benepar is installed
            self.benepar_helper.ensure_benepar_installed()
        
        # Load models
        try:
            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')
            
            # Setup Benepar in the spaCy pipeline with robust error handling
            if structured_logging:
                self.nlp = self.benepar_helper.setup_spacy_pipeline(self.nlp)
                # Download the Benepar model if needed
                self.benepar_helper.download_model()
            else:
                # Fallback method for adding Benepar
                try:
                    import benepar
                    if "benepar" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                except Exception as e:
                    print(f"Error setting up Benepar: {str(e)}")
            
            # Load BERT for token embeddings
            if structured_logging:
                app_logger.info("Loading BERT model and tokenizer...")
            self.bert = BertModel.from_pretrained(model_name).to(self.device)
            # Explicitly use fast tokenizer
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            if structured_logging:
                app_logger.info("BERT model and tokenizer loaded successfully")
            
            # Initialize Tree-LSTM models
            if structured_logging:
                app_logger.info("Initializing Tree-LSTM models...")
            
            if mode in ['bottom-up', 'both']:
                self.bottom_up_tree_lstm = ChildSumTreeLSTM(
                    d_in=768,   # BERT hidden size
                    d_hidden=768
                ).to(self.device)
            
            if mode in ['top-down', 'both']:
                self.top_down_tree_lstm = TopDownTreeLSTM(
                    d_in=768,   # BERT hidden size
                    d_hidden=768
                ).to(self.device)
            
            if structured_logging:
                app_logger.info("Tree-LSTM models initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            if structured_logging:
                app_logger.error(error_msg)
                import traceback
                app_logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def encode(self, sentence: str) -> Dict[str, Any]:
        """Process a sentence and return its tree-LSTM encoding."""
        if not sentence.strip():
            raise ValueError("Empty sentence provided")
        
        # Parse sentence
        doc = self.nlp(sentence)
        if not doc or len(list(doc.sents)) == 0:
            raise ValueError("Failed to parse sentence")
            
        # Get first sentence
        sent = list(doc.sents)[0]
        tokens = list(sent)
        
        # Get parse tree
        try:
            # Build our Tree node structure from constituency parse
            root = build_tree(sent)
            
            # Get BERT embeddings for all tokens
            with torch.no_grad():
                # Tokenize whole text
                encoded = self.tokenizer(
                    [token.text for token in tokens],
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Get BERT embeddings
                outputs = self.bert(**encoded)
                last_hidden = outputs.last_hidden_state.squeeze(0)
                
                # Map word pieces back to tokens
                token2vecs = {}
                for idx, word_idx in enumerate(encoded.word_ids()):
                    if word_idx is not None and word_idx < len(tokens):
                        if word_idx not in token2vecs:
                            token2vecs[word_idx] = []
                        token2vecs[word_idx].append(last_hidden[idx])
                
                # Average subword vectors for each token
                vec_lookup = {}
                for word_idx, vecs in token2vecs.items():
                    vec_lookup[word_idx] = torch.stack(vecs).mean(0)
            
            # Process through Tree-LSTM(s)
            with torch.no_grad():
                if self.mode == 'bottom-up' or self.mode == 'both':
                    bottom_up_embedding, _ = self.bottom_up_tree_lstm(root, vec_lookup)
                
                if self.mode == 'top-down' or self.mode == 'both':
                    top_down_embedding, _ = self.top_down_tree_lstm(root, vec_lookup)
                
                # Combine the embeddings if using both
                if self.mode == 'both':
                    # Combine bottom-up and top-down embeddings for each node
                    def combine_embeddings(node):
                        # Combine the representations
                        if node.h_bottom_up is not None and node.h_top_down is not None:
                            # Simple concatenation
                            # node.h = torch.cat([node.h_bottom_up, node.h_top_down], dim=0)
                            
                            # Element-wise addition (more efficient for downstream processing)
                            node.h = node.h_bottom_up + node.h_top_down
                            
                            # Element-wise multiplication (for gating effects)
                            # node.h = node.h_bottom_up * node.h_top_down
                            
                            # Cell state can be combined similarly
                            node.c = node.c_bottom_up + node.c_top_down
                        
                        # Process children recursively
                        for child in node.children:
                            combine_embeddings(child)
                    
                    # Apply the combination
                    combine_embeddings(root)
                    
                    # Root embedding is the combined embedding
                    root_embedding = root.h
                elif self.mode == 'bottom-up':
                    # Copy bottom-up embeddings to the main fields
                    def copy_bottom_up(node):
                        node.h = node.h_bottom_up
                        node.c = node.c_bottom_up
                        for child in node.children:
                            copy_bottom_up(child)
                    
                    copy_bottom_up(root)
                    root_embedding = bottom_up_embedding
                else:  # top-down
                    # Copy top-down embeddings to the main fields
                    def copy_top_down(node):
                        node.h = node.h_top_down
                        node.c = node.c_top_down
                        for child in node.children:
                            copy_top_down(child)
                    
                    copy_top_down(root)
                    root_embedding = top_down_embedding
        
        except Exception as e:
            error_msg = f"Error processing tree: {str(e)}"
            if structured_logging:
                app_logger.error(error_msg)
                import traceback
                app_logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg)
        
        return {
            "root_embedding": root_embedding.tolist(),
            "tree": root.to_dict(),
            "mode": self.mode
        } 