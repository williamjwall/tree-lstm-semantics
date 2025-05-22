import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

class Node:
    """Node class for representing tree structure."""
    def __init__(self, label=None, span=None, children=None):
        self.label = label
        self.span = span  # (start_idx, end_idx)
        self.children = children or []
        self.h = None  # Hidden state
        self.c = None  # Cell state
        self.f_gate = None  # For visualization
        
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (has no children)."""
        return len(self.children) == 0
    
    def __repr__(self):
        return f"Node({self.label}, {self.span})"
    
    def add_child(self, child):
        self.children.append(child)
        return self


class TopDownTreeLSTM(nn.Module):
    """
    Top-Down Tree-LSTM module that composes representations from parent to children.
    
    This implementation processes information from the root node down to the leaves,
    allowing information to flow from parent to children - the opposite of Child-Sum Tree-LSTM.
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
            
            # Store forget gate value for visualization
            node.f_gate = float(f.mean().item())
        
        # Store states in node
        node.h, node.c = h, c
        
        # Process all children recursively with current node as parent
        for child in node.children:
            self.forward(child, vec_lookup, h, c)
        
        return h, c


class TopDownTreeLSTMWithGates(nn.Module):
    """
    Top-Down Tree-LSTM with gate visualization for educational purposes.
    
    This implementation adds visualization capabilities to the top-down Tree-LSTM.
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
        Process the tree top-down with gate visualization.
        
        Args:
            node: Current node being processed
            vec_lookup: Dictionary mapping token indices to embeddings
            parent_h: Hidden state from parent node (None for root)
            parent_c: Cell state from parent node (None for root)
            
        Returns:
            Tuple of (h, c) tensors representing the hidden and cell states
        """
        # Initialize gate storage
        node.gates = {}
        node.h_tensor = None
        node.c_tensor = None
        
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
            
            # Store gate values for visualization
            node.gates['i_gate'] = i
            node.gates['o_gate'] = o
            node.gates['u_gate'] = u
            node.gates['f_gate'] = torch.ones_like(i) * 0.5  # Default for root
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
            
            # Store gate values for visualization
            node.gates['i_gate'] = i
            node.gates['o_gate'] = o
            node.gates['u_gate'] = u
            node.gates['f_gate'] = f
        
        # Store states in node
        node.h, node.c = h, c
        node.h_tensor = h
        node.c_tensor = c
        
        # Process all children recursively with current node as parent
        for child in node.children:
            self.forward(child, vec_lookup, h, c)
        
        return h, c


def run_example_top_down_treelstm(root: Node, vec_lookup: Dict[int, torch.Tensor], 
                                  d_in: int = 50, d_hidden: int = 50, 
                                  device: str = 'cpu') -> Node:
    """
    Run top-down Tree-LSTM on an example tree.
    
    Args:
        root: Root node of the tree
        vec_lookup: Dictionary mapping token indices to embeddings
        d_in: Input dimension
        d_hidden: Hidden dimension
        device: Device to run on
        
    Returns:
        Root node with updated hidden and cell states
    """
    treelstm = TopDownTreeLSTMWithGates(d_in=d_in, d_hidden=d_hidden).to(device)
    root_h, root_c = treelstm(root, vec_lookup)
    return root 