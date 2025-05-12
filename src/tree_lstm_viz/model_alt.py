import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

# Force CPU mode to avoid CUDA compatibility issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
    try:
        # If no sub-constituents, it's a leaf token
        children_spans = list(span._.children)
        if not children_spans:
            return Node("TOKEN", (span.start, span.end))

        # Otherwise, build child Nodes
        children = [build_tree(child) for child in children_spans]
        # Use the first label on this span (most trees have exactly one)
        label = span._.labels[0] if span._.labels else "UNKNOWN"
        return Node(label, (span.start, span.end), children)
    except Exception as e:
        # Simplified fallback for any parsing errors
        print(f"Error in build_tree: {e}")
        return Node("ERROR", (0, 1))

class ChildSumTreeLSTM(nn.Module):
    """
    Simplified Child-Sum Tree-LSTM module that composes representations bottom-up.
    This is a lightweight version designed to work on minimal compute resources.
    """
    def __init__(self, d_in=768, d_hidden=128):
        super().__init__()
        self.d_hidden = d_hidden
        
        # Input projection
        self.W_in = nn.Linear(d_in, d_hidden)
        
        # Cell update components
        self.W_update = nn.Linear(d_hidden, 4 * d_hidden)
    
    def forward(self, node: Node, vec_lookup: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified recursive processing of the tree starting from the given node.
        
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
                h = torch.zeros(self.d_hidden)
            else:
                # Project the token embedding to the right size
                h = self.W_in(vec_lookup[token_idx])
            
            # Initialize cell state to zeros
            c = torch.zeros_like(h)
            node.h, node.c = h, c
            return h, c
        
        # Internal node: process children recursively
        child_states = [self.forward(child, vec_lookup) for child in node.children]
        if not child_states:
            # Defensive - shouldn't happen but just in case
            h = c = torch.zeros(self.d_hidden)
            node.h, node.c = h, c
            return h, c
        
        # Stack hidden and cell states
        h_children = torch.stack([h for h, _ in child_states])
        c_children = torch.stack([c for _, c in child_states])
        
        # Average child hidden states
        h_avg = torch.mean(h_children, dim=0)
        
        # Compute gates (simplified)
        gates = self.W_update(h_avg)
        i, o, u, f = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        f = torch.sigmoid(f).unsqueeze(0)  # Use same forget gate for all children
        
        # Compute cell state (simple child-sum with same forget gate for all children)
        c = i * u + torch.sum(f * c_children, dim=0)
        
        # Compute hidden state
        h = o * torch.tanh(c)
        
        # Store states in node
        node.h, node.c = h, c
        return h, c

class TreeLSTMEncoder:
    """Simplified sentence encoder using Tree-LSTM over constituency parse trees."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        try:
            import spacy
            import benepar
            
            # Load NLP pipeline with parser
            self.nlp = spacy.load('en_core_web_sm')
            
            # Check if benepar is in the pipeline, add if not
            if 'benepar' not in self.nlp.pipe_names:
                # Try to get Berkeley parser, with fallback if not available
                try:
                    self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
                except:
                    print("Warning: Could not load benepar with model benepar_en3")
            
            # Initialize simplified embedding function using fixed random vectors
            # This is a fallback in case transformers aren't available
            self.embed_fn = self._random_embed
            
            # Try to load real BERT if available
            try:
                from transformers import BertModel, AutoTokenizer
                # Use CPU
                self.device = torch.device('cpu')
                # Load BERT for token embeddings
                self.bert = BertModel.from_pretrained(model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Switch to real embedding function
                self.embed_fn = self._bert_embed
                print("Using BERT for embeddings")
            except Exception as e:
                print(f"Falling back to random embeddings: {e}")
            
            # Initialize Tree-LSTM (smaller than original to conserve memory)
            self.tree_lstm = ChildSumTreeLSTM(d_in=768, d_hidden=128)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TreeLSTMEncoder: {str(e)}")
    
    def _random_embed(self, tokens):
        """Fallback embedding function using fixed random vectors."""
        np.random.seed(42)  # For consistent random embeddings
        embs = {}
        for i, token in enumerate(tokens):
            # Create a random but consistent vector for each token
            embs[i] = torch.tensor(np.random.normal(0, 1, 768).astype(np.float32))
        return embs
    
    def _bert_embed(self, tokens):
        """Get BERT embeddings for tokens."""
        embs = {}
        try:
            token_texts = [token.text for token in tokens]
            
            # Process in batches of 1 token to avoid memory issues
            for i, text in enumerate(token_texts):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.bert(**inputs)
                # Use mean of token embeddings (excluding special tokens)
                embs[i] = outputs.last_hidden_state[0, 1:-1].mean(dim=0)
        except Exception as e:
            print(f"Error in BERT embedding: {e}, falling back to random")
            embs = self._random_embed(tokens)
        
        return embs
    
    def encode(self, sentence: str) -> Dict[str, Any]:
        """Process a sentence and return its tree-LSTM encoding."""
        if not sentence.strip():
            raise ValueError("Empty sentence provided")
        
        try:
            # Parse sentence
            doc = self.nlp(sentence)
            if not doc or not list(doc.sents):
                raise ValueError("Failed to parse sentence")
                
            # Get first sentence & tokens
            sent = list(doc.sents)[0]
            tokens = list(sent)
            
            # Build tree
            root = build_tree(sent)
            
            # Get embeddings for tokens
            vec_lookup = self.embed_fn(tokens)
            
            # Process through Tree-LSTM
            with torch.no_grad():
                root_embedding, _ = self.tree_lstm(root, vec_lookup)
            
            return {
                "root_embedding": root_embedding.tolist(),
                "tree": root.to_dict()
            }
        except Exception as e:
            raise RuntimeError(f"Error processing sentence: {str(e)}") 