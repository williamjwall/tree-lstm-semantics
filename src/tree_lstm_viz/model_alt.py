import torch
import torch.nn as nn
import spacy
from typing import List, Dict, Any, Optional, Tuple

# Force CPU mode to avoid CUDA compatibility issues
torch.cuda.is_available = lambda: False

class Node:
    """Tree node for constituency parsing tree."""
    def __init__(self, label: str, span: tuple, children: List['Node'] = None):
        self.label = label
        self.span = span  # (start_idx, end_idx) in spaCy tokens
        self.children = children or []
        self.h = self.c = None  # hidden and cell states for Tree-LSTM
    
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
        Recursively process the tree starting from the given node.
        
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
            node.h, node.c = h, c
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
            # Optionally store forget gate value for visualization
            child.f_gate = float(f_k.mean().item())
        
        # Compute hidden state from cell
        h = o * torch.tanh(c)
        
        # Store states in node
        node.h, node.c = h, c
        return h, c

class TreeLSTMEncoder:
    """Encode sentences using Tree-LSTM over constituency parse trees."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        import benepar
        from transformers import BertModel, BertTokenizer
        
        # Set device (CPU only for compatibility)
        self.device = torch.device('cpu')
        
        # Load models
        try:
            self.nlp = spacy.load('en_core_web_sm')
            # Add Benepar component for constituency parsing
            self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
            
            # Load BERT for token embeddings
            self.bert = BertModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
            # Initialize Tree-LSTM
            self.tree_lstm = ChildSumTreeLSTM(
                d_in=768,   # BERT hidden size
                d_hidden=768
            ).to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
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
                # Process each token separately and track offsets
                vec_lookup = {}
                for i, token in enumerate(tokens):
                    # Tokenize this token
                    token_encoding = self.tokenizer(token.text, return_tensors="pt", 
                                                  add_special_tokens=False).to(self.device)
                    
                    # Get embeddings for this token's subwords
                    token_output = self.bert(**token_encoding)
                    token_embeddings = token_output.last_hidden_state.squeeze(0)
                    
                    # Average the subword embeddings for this token
                    token_embedding = token_embeddings.mean(dim=0)
                    vec_lookup[i] = token_embedding
            
            # Process through Tree-LSTM
            with torch.no_grad():
                root_embedding, _ = self.tree_lstm(root, vec_lookup)
        
        except Exception as e:
            raise RuntimeError(f"Error processing tree: {str(e)}")
        
        return {
            "root_embedding": root_embedding.tolist(),
            "tree": root.to_dict()
        } 