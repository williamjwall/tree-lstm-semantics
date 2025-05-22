#!/usr/bin/env python3

# Tree-LSTM Streamlit Interface for Educational Purposes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import time
import copy
# Import top-down TreeLSTM
from src.tree_lstm_viz.top_down_model import TopDownTreeLSTM, TopDownTreeLSTMWithGates, run_example_top_down_treelstm

# Set page configuration
st.set_page_config(
    page_title="Tree-LSTM Learning Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful interface
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4895ef !important;
        color: white !important;
    }
    h1 {
        color: #3a0ca3;
        font-weight: 800;
    }
    h2, h3 {
        color: #4895ef;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #4895ef;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a0ca3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e9ecef;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #4895ef;
    }
    .math-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #b5179e;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .node-highlight {
        border: 2px solid #4895ef;
        border-radius: 8px;
        padding: 15px;
        background-color: #f0f4ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #b5179e;
    }
    .metric-label {
        font-size: 16px;
        font-weight: 600;
        color: #4895ef;
        margin-top: 5px;
    }
    /* LaTeX styling */
    .katex {
        font-size: 1.1em !important;
    }
    /* Link styling */
    a {
        color: #4895ef;
        text-decoration: none;
    }
    a:hover {
        color: #b5179e;
        text-decoration: underline;
    }
    /* Code blocks */
    code {
        background-color: #f8f9fa;
        padding: 2px 5px;
        border-radius: 4px;
        color: #b5179e;
    }
    /* Divider styling */
    hr {
        border-top: 1px solid #e9ecef;
        margin: 30px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Define Node class for tree structure
class Node:
    def __init__(self, label=None, span=None, children=None):
        self.label = label
        self.span = span  # (start_idx, end_idx)
        self.children = children or []
        
    def __repr__(self):
        return f"Node({self.label}, {self.span})"
    
    def add_child(self, child):
        self.children.append(child)
        return self

# Child-Sum Tree-LSTM Implementation
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, d_in, d_hidden):
        """
        Initialize the Child-Sum Tree-LSTM model
        
        Args:
            d_in: Dimension of input vectors
            d_hidden: Dimension of hidden/cell states
        """
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        
        # Input, output, update gates (combined for efficiency)
        self.W_iou = nn.Linear(d_in, 3 * d_hidden)  # Maps input to [i, o, u] gates
        self.U_iou = nn.Linear(d_hidden, 3 * d_hidden, bias=False)  # Maps sum of children to [i, o, u] gates
        
        # Forget gate - one per child
        self.W_f = nn.Linear(d_in, d_hidden)  # Maps input to forget gate
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)  # Maps each child to its forget gate
    
    def forward(self, node, vec_lookup=None):
        """
        Forward pass through the Tree-LSTM
        
        Args:
            node: Current tree node being processed
            vec_lookup: Dictionary mapping token indices to input vectors
            
        Returns:
            h, c: Hidden state and cell state for the current node
        """
        if not node.children:  # Leaf node
            if vec_lookup is not None:
                x = vec_lookup[node.span[0]]  # Use pre-computed embedding
            else:
                # Initialize with random if no embedding provided
                x = torch.randn(self.d_in, device=next(self.parameters()).device)
                
            h, c = self._node_forward(x, [], [])
            return h, c
        
        # Process all children recursively
        child_h = []
        child_c = []
        
        for child in node.children:
            h_j, c_j = self.forward(child, vec_lookup)
            child_h.append(h_j)
            child_c.append(c_j)
        
        # Use the span's first token as input for the current node
        if vec_lookup is not None:
            x = vec_lookup[node.span[0]]
        else:
            x = torch.randn(self.d_in, device=next(self.parameters()).device)
            
        h, c = self._node_forward(x, child_h, child_c)
        return h, c
    
    def _node_forward(self, x, child_h, child_c):
        """
        Process a single node in the Tree-LSTM
        
        Args:
            x: Input vector for the current node
            child_h: List of hidden states from children
            child_c: List of cell states from children
            
        Returns:
            h, c: Hidden state and cell state for the current node
        """
        # If this is a leaf node with no children
        if not child_h:
            # Compute input, output and update gates
            iou = self.W_iou(x)
            i, o, u = torch.split(iou, self.d_hidden, dim=-1)
            i = torch.sigmoid(i)  # Input gate: controls new information
            o = torch.sigmoid(o)  # Output gate: controls output exposure
            u = torch.tanh(u)     # Update gate: creates candidate values
            
            # For leaf nodes, cell state is just input gate * update gate
            c = i * u
            
            # Hidden state is output gate * tanh(cell state)
            h = o * torch.tanh(c)
            return h, c
        
        # Calculate sum of children's hidden states (key part of Child-Sum Tree-LSTM)
        h_sum = sum(child_h)
        
        # Input, output, update gates
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = torch.split(iou, self.d_hidden, dim=-1)
        i = torch.sigmoid(i)  # Input gate
        o = torch.sigmoid(o)  # Output gate
        u = torch.tanh(u)     # Update gate
        
        # Calculate forget gate for each child
        f = [torch.sigmoid(self.W_f(x) + self.U_f(h_j)) for h_j in child_h]
        
        # Calculate cell state: new info + selective memory from children
        c = i * u + sum(f_j * c_j for f_j, c_j in zip(f, child_c))
        
        # Calculate hidden state
        h = o * torch.tanh(c)
        
        return h, c

# Tree-LSTM with Visible Gates for educational purposes
class ChildSumTreeLSTMWithGates(nn.Module):
    def __init__(self, d_in, d_hidden):
        """
        Initialize the Child-Sum Tree-LSTM model with gate visualization
        
        Args:
            d_in: Dimension of input vectors
            d_hidden: Dimension of hidden/cell states
        """
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        
        # Input, output, update gates
        self.W_iou = nn.Linear(d_in, 3 * d_hidden)
        self.U_iou = nn.Linear(d_hidden, 3 * d_hidden, bias=False)
        
        # Forget gate - one per child
        self.W_f = nn.Linear(d_in, d_hidden)
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)
    
    def forward(self, node, vec_lookup=None):
        """
        Forward pass through the Tree-LSTM with gate visualization
        
        Args:
            node: Current tree node being processed
            vec_lookup: Dictionary mapping token indices to input vectors
            
        Returns:
            h, c: Hidden state and cell state for the current node
        """
        # Store gate values for visualization
        node.gates = {}
        node.h_tensor = None
        node.c_tensor = None
        
        if not node.children:  # Leaf node
            if vec_lookup is not None:
                x = vec_lookup[node.span[0]]  # Use pre-computed embedding
            else:
                # Initialize with random if no embedding provided
                x = torch.randn(self.d_in, device=next(self.parameters()).device)
                
            h, c = self._node_forward(x, [], [], node)
            node.h_tensor = h
            node.c_tensor = c
            return h, c
        
        # Process all children recursively
        child_h = []
        child_c = []
        
        for child in node.children:
            h_j, c_j = self.forward(child, vec_lookup)
            child_h.append(h_j)
            child_c.append(c_j)
        
        # Use the span's first token as input for the current node
        if vec_lookup is not None:
            x = vec_lookup[node.span[0]]
        else:
            x = torch.randn(self.d_in, device=next(self.parameters()).device)
            
        h, c = self._node_forward(x, child_h, child_c, node)
        node.h_tensor = h
        node.c_tensor = c
        return h, c
    
    def _node_forward(self, x, child_h, child_c, node):
        """
        Process a single node in the Tree-LSTM with gate visualization
        
        Args:
            x: Input vector for the current node
            child_h: List of hidden states from children
            child_c: List of cell states from children
            node: Current tree node for storing gate values
            
        Returns:
            h, c: Hidden state and cell state for the current node
        """
        # If this is a leaf node with no children
        if not child_h:
            iou = self.W_iou(x)
            i, o, u = torch.split(iou, self.d_hidden, dim=-1)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            c = i * u
            h = o * torch.tanh(c)
            
            # Store gate values
            node.gates['i_gate'] = i
            node.gates['o_gate'] = o
            node.gates['u_gate'] = u
            node.gates['f_gate'] = torch.ones_like(i) * 0.5  # Default for leaf nodes
            
            return h, c
        
        # Calculate sum of children's hidden states
        h_sum = sum(child_h)
        
        # Input, output, update gates
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = torch.split(iou, self.d_hidden, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        
        # Calculate forget gate for each child
        f = [torch.sigmoid(self.W_f(x) + self.U_f(h_j)) for h_j in child_h]
        
        # Calculate cell state
        c = i * u + sum(f_j * c_j for f_j, c_j in zip(f, child_c))
        
        # Calculate hidden state
        h = o * torch.tanh(c)
        
        # Store gate values
        node.gates['i_gate'] = i
        node.gates['o_gate'] = o
        node.gates['u_gate'] = u
        if f:
            # Average forget gates for visualization
            node.gates['f_gate'] = torch.stack(f).mean(dim=0)
        else:
            node.gates['f_gate'] = torch.ones_like(i) * 0.5
        
        return h, c

# Create a simple example tree for demonstration
def create_example_tree():
    """Create a simple tree for educational demonstration"""
    # Create nodes
    root = Node(label="S", span=(0, 4))  # Sentence
    child1 = Node(label="NP", span=(0, 2))  # Noun Phrase
    child2 = Node(label="VP", span=(2, 4))  # Verb Phrase
    grandchild1 = Node(label="DT", span=(0, 1))  # Determiner (The)
    grandchild2 = Node(label="NN", span=(1, 2))  # Noun (cat)
    grandchild3 = Node(label="VBZ", span=(2, 3))  # Verb (is)
    grandchild4 = Node(label="JJ", span=(3, 4))  # Adjective (fat)
    
    # Build tree structure
    child1.add_child(grandchild1)
    child1.add_child(grandchild2)
    child2.add_child(grandchild3)
    child2.add_child(grandchild4)
    root.add_child(child1)
    root.add_child(child2)
    
    return root

# Function to visualize the tree with Plotly
def visualize_tree(root):
    G = nx.DiGraph()
    node_labels = {}
    node_colors = {}
    
    # Color mapping for different node types - using brighter, more vibrant colors
    color_map = {
        'S': '#4895ef',   # Sentence - bright blue
        'NP': '#b5179e',  # Noun Phrase - magenta
        'VP': '#480ca8',  # Verb Phrase - deep purple
        'DT': '#f72585',  # Determiner - pink
        'NN': '#3a0ca3',  # Noun - deep blue
        'VB': '#4cc9f0',  # Verb - light blue
        'PP': '#7209b7',  # Preposition - purple
        'Root': '#4895ef' # Default - bright blue
    }
    
    # Store node positions manually
    positions = {}
    levels = {}
    
    def add_nodes_edges(node, parent_id=None, depth=0, pos_idx=0):
        # Create node ID
        node_id = f"{node.label}_{node.span[0]}_{node.span[1]}"
        
        # Track node depth
        levels[node_id] = depth
        
        # Node label
        label = f"{node.label}\n{node.span}"
        
        # Add node
        G.add_node(node_id, depth=depth, label=node.label, span=node.span)
        node_labels[node_id] = label
        
        # Assign color based on label
        if node.label in color_map:
            node_colors[node_id] = color_map[node.label]
        else:
            node_colors[node_id] = color_map['Root']
        
        # Add edge from parent
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Process children
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            # Calculate position index for child
            child_pos = pos_idx - (child_count - 1) / 2 + i
            add_nodes_edges(child, node_id, depth+1, child_pos)
    
    # Build graph
    add_nodes_edges(root)
    
    # Create manual positions for nodes based on tree structure
    max_depth = max(levels.values()) if levels else 0
    width_unit = 1.0
    
    for node_id, level in levels.items():
        # Get all nodes at this level
        nodes_at_level = [n for n, l in levels.items() if l == level]
        # Find position of this node in its level
        pos_in_level = nodes_at_level.index(node_id)
        # Calculate x position
        x_pos = (pos_in_level - (len(nodes_at_level) - 1) / 2) * width_unit * (max_depth - level + 1)
        # Calculate y position (top to bottom)
        y_pos = -level * 1.5
        
        positions[node_id] = (x_pos, y_pos)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges as lines
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'))
    
    # Add nodes as markers
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_labels_list = []
    
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels[node])
        node_color.append(node_colors[node])
        node_labels_list.append(G.nodes[node]['label'])
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=40,
            line=dict(width=2, color='white')
        ),
        text=node_labels_list,
        textposition="middle center",
        textfont=dict(color='white', size=14, family="Arial, sans-serif"),
        hovertext=node_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        width=700
    )
    
    return fig

# Function to create heatmap for vector visualization using Plotly
def create_heatmap(vector, title="Vector Heatmap"):
    if vector is None:
        return None
    
    # Convert to numpy and reshape for heatmap
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().cpu().numpy()
    
    # Reshape to 2D grid (approximately square)
    size = vector.shape[0]
    width = int(np.sqrt(size))
    height = (size + width - 1) // width  # Ceiling division
    
    # Pad the vector if needed
    padded = np.zeros(width * height)
    padded[:size] = vector
    
    # Reshape to 2D
    grid = padded.reshape(height, width)
    
    # Create heatmap with Plotly
    fig = px.imshow(
        grid,
        color_continuous_scale=['#3a0ca3', '#f8f9fa', '#f72585'],
        labels=dict(color="Value"),
        title=title
    )
    
    fig.update_layout(
        height=400,
        width=600,
        coloraxis_colorbar=dict(
            title="Value",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

# Function to create bar chart for gate values using Plotly
def create_gate_chart(node, title="Gate Values"):
    if not hasattr(node, 'gates'):
        return None
    
    # Get mean gate values
    gate_means = {
        'Input Gate': node.gates['i_gate'].mean().item(),
        'Output Gate': node.gates['o_gate'].mean().item(),
        'Update Gate': node.gates['u_gate'].mean().item(),
        'Forget Gate': node.gates['f_gate'].mean().item()
    }
    
    # Colors for gates - updated to match new color scheme
    colors = ['#f72585', '#4895ef', '#4cc9f0', '#b5179e']
    
    # Create bar chart with Plotly
    fig = go.Figure()
    
    for i, (gate, value) in enumerate(gate_means.items()):
        fig.add_trace(go.Bar(
            x=[gate],
            y=[value],
            name=gate,
            marker_color=colors[i],
            text=[f'{value:.2f}'],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=title,
        yaxis=dict(
            title='Gate Value',
            range=[0, 1]
        ),
        height=400,
        width=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Function to create a radar chart for gate values
def create_gate_radar(node, title="Gate Values"):
    if not hasattr(node, 'gates'):
        return None
    
    # Get mean gate values
    gate_means = {
        'Input Gate': node.gates['i_gate'].mean().item(),
        'Output Gate': node.gates['o_gate'].mean().item(),
        'Update Gate': node.gates['u_gate'].mean().item(),
        'Forget Gate': node.gates['f_gate'].mean().item()
    }
    
    # Create radar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(gate_means.values()),
        theta=list(gate_means.keys()),
        fill='toself',
        line=dict(color='#4895ef', width=3),
        fillcolor='rgba(72, 149, 239, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor='#f8f9fa'
        ),
        title=title,
        height=400,
        width=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Function to get all nodes in a tree
def get_all_nodes(root):
    nodes = [root]
    for child in root.children:
        nodes.extend(get_all_nodes(child))
    return nodes

# Function to get node by ID
def get_node_by_id(root, node_id):
    all_nodes = get_all_nodes(root)
    for node in all_nodes:
        curr_id = f"{node.label}_{node.span[0]}_{node.span[1]}"
        if curr_id == node_id:
            return node
    return None

# Function to get node IDs for dropdown
def get_node_ids_with_labels(root):
    all_nodes = get_all_nodes(root)
    return {f"{node.label}_{node.span[0]}_{node.span[1]}": f"{node.label} {node.span}" 
            for node in all_nodes}

# Quiz questions
quiz_questions = [
    {
        "question": "What is the role of the forget gate in a Tree-LSTM?",
        "options": [
            "It determines how much information to forget from the current input",
            "It determines how much information to forget from each child's cell state",
            "It determines how much information to output to the next node",
            "It determines how much information to update in the cell state"
        ],
        "answer": 1,  # B
        "explanation": "The forget gate in a Tree-LSTM produces a value between 0 and 1 for each child's cell state, determining how much information to keep from each child."
    },
    {
        "question": "In a Child-Sum Tree-LSTM, how are the hidden states of children combined?",
        "options": [
            "They are concatenated",
            "They are averaged",
            "They are summed",
            "They are multiplied element-wise"
        ],
        "answer": 2,  # C
        "explanation": "In a Child-Sum Tree-LSTM, the hidden states of all children are summed together before being used in the parent's computation, which is a key characteristic of this architecture."
    },
    {
        "question": "What is the key difference between a standard LSTM and a Tree-LSTM?",
        "options": [
            "Tree-LSTMs use different activation functions",
            "Tree-LSTMs can handle multiple children for each node",
            "Tree-LSTMs don't use cell states",
            "Tree-LSTMs have fewer parameters"
        ],
        "answer": 1,  # B
        "explanation": "While standard LSTMs process sequential data with one input and one previous state, Tree-LSTMs can handle tree-structured data where each node can have multiple children."
    },
    {
        "question": "In the gradient flow of a Tree-LSTM, information can flow:",
        "options": [
            "Only from parent to children",
            "Only from children to parent",
            "Both from parent to children and from children to parent",
            "Neither from parent to children nor from children to parent"
        ],
        "answer": 2,  # C
        "explanation": "During backpropagation, gradients flow both from parent to children and from children to parent, allowing the model to learn from the entire tree structure."
    },
    {
        "question": "The input gate in a Tree-LSTM controls:",
        "options": [
            "How much new information to add to the cell state",
            "How much information to pass to the children",
            "How much information to receive from the children",
            "How much information to output from the node"
        ],
        "answer": 0,  # A
        "explanation": "The input gate controls how much of the new candidate values (created by the update gate) will be added to the cell state."
    },
    {
        "question": "What is the primary advantage of using a Tree-LSTM over a standard LSTM?",
        "options": [
            "Tree-LSTMs are faster to train.",
            "Tree-LSTMs can handle hierarchical data structures.",
            "Tree-LSTMs require less data for training.",
            "Tree-LSTMs have fewer parameters."
        ],
        "answer": 1,  # B
        "explanation": "Tree-LSTMs are specifically designed to handle hierarchical data structures, making them suitable for tasks like parsing and sentiment analysis."
    },
    {
        "question": "Which gate in a Tree-LSTM is responsible for deciding how much information to pass to the parent node?",
        "options": [
            "Input Gate",
            "Forget Gate",
            "Output Gate",
            "Update Gate"
        ],
        "answer": 2,  # C
        "explanation": "The output gate controls how much of the cell state is exposed as the hidden state, which is passed to the parent node."
    },
    {
        "question": "In what scenario would a Tree-LSTM be preferred over a Transformer model?",
        "options": [
            "When processing sequential data.",
            "When dealing with large datasets.",
            "When interpretability of the model is important.",
            "When computational resources are unlimited."
        ],
        "answer": 2,  # C
        "explanation": "Tree-LSTMs are preferred when interpretability is important, as they explicitly model hierarchical structures."
    },
    {
        "question": "How does a Tree-LSTM handle multiple children nodes?",
        "options": [
            "By concatenating their hidden states.",
            "By averaging their hidden states.",
            "By summing their hidden states.",
            "By ignoring all but one child."
        ],
        "answer": 2,  # C
        "explanation": "A Tree-LSTM sums the hidden states of all children nodes, which is a key feature of the Child-Sum Tree-LSTM architecture."
    },
    {
        "question": "What is a common application of Tree-LSTMs in natural language processing?",
        "options": [
            "Image classification",
            "Speech recognition",
            "Sentiment analysis",
            "Time series forecasting"
        ],
        "answer": 2,  # C
        "explanation": "Tree-LSTMs are commonly used in sentiment analysis, especially when dealing with sentences that have complex hierarchical structures."
    }
]

def check_quiz_answers(user_answers):
    correct = 0
    feedback = []
    
    for i, (q, a) in enumerate(zip(quiz_questions, user_answers)):
        if a == q["answer"]:
            correct += 1
            feedback.append(f"Q{i+1}: Correct! {q['explanation']}")
        else:
            feedback.append(f"Q{i+1}: Incorrect. The correct answer is: {q['options'][q['answer']]}. {q['explanation']}")
    
    score = f"Score: {correct}/{len(quiz_questions)}"
    return score, feedback

# Run example Tree-LSTM and return results
def run_example_treelstm(device='cpu', d_in=50, d_hidden=50):
    # Create example tree
    root = create_example_tree()
    
    # Create word embeddings for "The cat is fat"
    vec_lookup = {
        0: torch.tensor([0.2, -0.5, 0.1, 0.3, 0.2] + [0.0] * (d_in-5), device=device),  # The
        1: torch.tensor([0.7, 0.3, 0.5, -0.2, 0.1] + [0.0] * (d_in-5), device=device),  # cat
        2: torch.tensor([0.1, 0.1, 0.4, 0.2, -0.3] + [0.0] * (d_in-5), device=device),  # is
        3: torch.tensor([0.5, 0.8, -0.2, 0.4, 0.3] + [0.0] * (d_in-5), device=device)   # fat
    }
    
    # Run Tree-LSTM
    treelstm = ChildSumTreeLSTMWithGates(d_in=d_in, d_hidden=d_hidden).to(device)
    root_vec, _ = treelstm(root, vec_lookup)
    
    return root

# After the run_example_treelstm function, add the top-down example runner function
def run_example_both_treelstm(device='cpu', d_in=50, d_hidden=50):
    """Run both bottom-up and top-down Tree-LSTM on the same example tree"""
    # Create example tree
    root = create_example_tree()
    
    # Create a copy for top-down model
    root_topdown = copy.deepcopy(root)
    
    # Create word embeddings for "The cat is fat"
    vec_lookup = {
        0: torch.tensor([0.2, -0.5, 0.1, 0.3, 0.2] + [0.0] * (d_in-5), device=device),  # The
        1: torch.tensor([0.7, 0.3, 0.5, -0.2, 0.1] + [0.0] * (d_in-5), device=device),  # cat
        2: torch.tensor([0.1, 0.1, 0.4, 0.2, -0.3] + [0.0] * (d_in-5), device=device),  # is
        3: torch.tensor([0.5, 0.8, -0.2, 0.4, 0.3] + [0.0] * (d_in-5), device=device)   # fat
    }
    
    # Run bottom-up Tree-LSTM
    bottomup_treelstm = ChildSumTreeLSTMWithGates(d_in=d_in, d_hidden=d_hidden).to(device)
    root_vec_bottomup, _ = bottomup_treelstm(root, vec_lookup)
    
    # Run top-down Tree-LSTM
    topdown_treelstm = TopDownTreeLSTMWithGates(d_in=d_in, d_hidden=d_hidden).to(device)
    root_vec_topdown, _ = topdown_treelstm(root_topdown, vec_lookup)
    
    return root, root_topdown

# Main app layout
def main():
    # Apply custom CSS
    local_css()
    
    # Title and introduction
    st.title("Understanding Tree-LSTM Networks")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Interactive Tree-LSTM", "Bottom-Up vs Top-Down", "Applications", "Resources"])

    # Tab 1: Introduction
    with tab1:
        st.header("Introduction to Tree-Structured LSTMs")
        
        st.markdown("""
        Tree-LSTM extends the standard LSTM to tree-structured data, allowing it to capture 
        hierarchical relationships in a more natural way than sequential models.
        
        This interactive app will help you understand:
        
        * How Tree-LSTMs differ from regular LSTMs
        * How information flows through a tree structure
        * The role of gates in Tree-LSTM
        * Practical applications of Tree-LSTMs
        """)
        
        # Show example sentence tree
        st.subheader("Example: A Simple Parse Tree")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            # Display the parse tree
            tree_example = """
            Sentence: "The cat is fat"
            
            Parse Tree:
                    S
                   / \\
                  /   \\
                NP     VP
               / \\    / \\
             DT   NN VBZ  JJ
             |    |   |   |
            The  cat  is  fat
            """
            st.code(tree_example, language="text")
        
        with col3:
            st.markdown("""
            **Tree Structure:**
            
            * **S**: Sentence
            * **NP**: Noun Phrase
            * **VP**: Verb Phrase
            * **DT**: Determiner
            * **NN**: Noun
            * **VBZ**: Verb
            * **JJ**: Adjective
            
            The tree reflects the grammatical structure of the sentence, making 
            it possible to capture compositional meaning through the hierarchy.
            """)
        
        # Compare Standard LSTM vs Tree-LSTM
        st.subheader("Standard LSTM vs. Tree-LSTM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Standard LSTM (Sequential)")
            st.markdown(r"""
            ```
            Input: [x₁, x₂, x₃, x₄]
            
            x₁ → LSTM → h₁
                  ↓
            x₂ → LSTM → h₂
                  ↓
            x₃ → LSTM → h₃
                  ↓
            x₄ → LSTM → h₄
            ```
            
            * Processes words one after another
            * Hidden state flows linearly from one word to the next
            * Single forget gate forgets previous hidden state
            * Cannot capture hierarchical structure
            """)
        
        with col2:
            st.markdown("### Tree-LSTM (Hierarchical)")
            st.markdown(r"""
            ```
                     [ROOT]
                    /      \
                   /        \
                  ○          ○
                 / \        / \
                /   \      /   \
               ○     ○    ○     ○
              / \   / \  / \   / \
             x₁  x₂ x₃ x₄ x₅  x₆ x₇ x₈
            ```
            
            * Processes words based on grammatical structure
            * Hidden states flow up from children to parents
            * One forget gate per child
            * Naturally captures hierarchical relationships
            """)
        
        # Add comparison table with key differences
        st.markdown("### Key Architectural Differences")
        
        comparison_data = {
            'Feature': ['Input Processing', 'Memory Flow', 'Forget Gates', 'Suited For', 'Representation Power', 'Training Complexity'],
            'Standard LSTM': ['Sequential (one after another)', 'Linear (previous → current)', '1 gate for previous state', 'Sequences (text, time series)', 'Limited for hierarchical data', 'Simpler, more common'],
            'Tree-LSTM': ['Hierarchical (bottom-up)', 'Tree-structured (children → parent)', '1 gate per child node', 'Trees (parse trees, XML)', 'Better for compositional data', 'More complex, specialized']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Add an interactive example showing compositional semantics
        st.subheader("Why Tree Structure Matters: Compositional Semantics")
        
        st.markdown(r"""
        Consider how the meaning changes in these sentences with the same words:
        
        **Sentence 1**: "The dog bit the man"
        ```
                S
               / \
              NP  VP
             / \  / \
           The dog bit NP
                     / \
                   the man
        ```
        
        **Sentence 2**: "The man bit the dog"
        ```
                S
               / \
              NP  VP
             / \  / \
           The man bit NP
                     / \
                   the dog
        ```
        
        Though they contain identical words, the meaning is entirely different. Tree-LSTMs can capture this difference because they process the words according to the grammatical structure, while standard LSTMs would process both sentences similarly.
        """)

    # Tab 2: Interactive Tree-LSTM
    with tab2:
        st.header("Tree-LSTM Interactive Visualization")
        
        st.markdown("""
        Here you can interact with a Child-Sum Tree-LSTM model to see how it processes a simple parsed sentence.
        
        The model processes our example sentence "The cat is fat" according to its parse tree structure.
        """)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Model Parameters")
            input_dim = st.slider("Input Dimension", min_value=10, max_value=100, value=50, step=10, 
                                 help="Size of word embeddings (larger = more expressive)")
            hidden_dim = st.slider("Hidden Dimension", min_value=10, max_value=100, value=50, step=10,
                                  help="Size of hidden states (larger = more capacity)")
            
            run_button = st.button("Run Tree-LSTM", help="Process the sentence with Tree-LSTM")
            
            if "tree_root" in st.session_state and run_button:
                st.session_state.pop("tree_root")
                
            if run_button or "tree_root" not in st.session_state:
                with st.spinner("Processing 'The cat is fat' with Tree-LSTM..."):
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    root = run_example_treelstm(device=device, d_in=input_dim, d_hidden=hidden_dim)
                    st.session_state.tree_root = root
            
            # Add explanation of the gate dynamics
            if "tree_root" in st.session_state:
                st.markdown("""
                ### Gate Dynamics Explained
                
                Each node in the tree uses gates to control information flow:
                
                1. **Input & Update Gates** work together to determine what new information to store
                2. **Forget Gates** (one per child) determine what information to keep from each child
                3. **Output Gate** determines what information to expose to the parent node
                
                Higher gate values (closer to 1) mean more information passes through.
                """)
        
        with col2:
            # Only show tree if we have processed it
            if "tree_root" in st.session_state:
                root = st.session_state.tree_root
                
                # Create visualization
                tab_vis1, tab_vis2 = st.tabs(["Tree Visualization", "Gate Values"])
                
                with tab_vis1:
                    fig = visualize_tree(root, show_gates=False)
                    st.pyplot(fig)
                
                with tab_vis2:
                    gate_fig = visualize_tree(root, show_gates=True)
                    st.pyplot(gate_fig)
                    
                    st.markdown("""
                    **Gate Visualization Legend:**
                    * **Red circle**: Input gate - controls new information flow
                    * **Green circle**: Output gate - controls output exposure
                    * **Blue circle**: Update gate - creates candidate values
                    * **Purple circle**: Forget gate - controls information from children
                    
                    The size of each circle represents the average activation value of that gate (larger = higher activation).
                    """)
            else:
                st.info("Click 'Run Tree-LSTM' to visualize the processing of the parse tree.")
                
                # Show what will be processed
                st.markdown("### Parse Tree Structure")
                
                # Show the static tree before processing
                blank_root = create_example_tree()
                blank_fig = visualize_tree(blank_root, show_gates=False, basic=True)
                st.pyplot(blank_fig)

    # Tab 3: Bottom-Up vs Top-Down
    with tab3:
        st.header("Bottom-Up vs Top-Down Tree-LSTM")
        
        st.markdown("""
        Tree-LSTM can process trees in two different ways:
        
        1. **Bottom-Up** (Child-Sum Tree-LSTM): Information flows from leaves to root
        2. **Top-Down** Tree-LSTM: Information flows from root to leaves
        
        Let's compare these approaches and see how they differ:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Bottom-Up Tree-LSTM")
            st.markdown(r"""
            ```
                     [ROOT] ⬆️
                    /      \
                   /        \
                  ○          ○
                 / \        / \
                /   \      /   \
               ○     ○    ○     ○
              
             x₁  x₂ x₃ x₄ 
            ```
            
            * Information flows from leaves to root
            * Child nodes influence parent nodes
            * Useful for compositional meaning
            * Children are processed before parents
            """)
        
        with col2:
            st.markdown("### Top-Down Tree-LSTM")
            st.markdown(r"""
            ```
                     [ROOT] ⬇️
                    /      \
                   /        \
                  ○          ○
                 / \        / \
                /   \      /   \
               ○     ○    ○     ○
              
             x₁  x₂ x₃ x₄ 
            ```
            
            * Information flows from root to leaves
            * Parent nodes influence child nodes
            * Useful for contextual meaning
            * Parents are processed before children
            """)
        
        # Add comparison table
        st.markdown("### Key Differences")
        
        comparison_data = {
            'Feature': ['Information Flow', 'Forget Gate Purpose', 'Processing Order', 'Best For', 'Contextual Understanding'],
            'Bottom-Up': ['Leaves to Root', 'Select information from children', 'Children before parents', 'Composing meaning from parts', 'Local context only'],
            'Top-Down': ['Root to Leaves', 'Select information from parent', 'Parents before children', 'Distributing context to parts', 'Global context to local elements']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Interactive visualization
        st.markdown("### Interactive Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Parameters")
            input_dim = st.slider("Input Dimension", min_value=10, max_value=100, value=50, step=10, 
                                 help="Size of word embeddings (larger = more expressive)")
            hidden_dim = st.slider("Hidden Dimension", min_value=10, max_value=100, value=50, step=10,
                                  help="Size of hidden states (larger = more capacity)")
            
            compare_button = st.button("Compare Both Models", help="Process the sentence with both Tree-LSTM approaches")
            
            if "bottomup_tree" in st.session_state and compare_button:
                st.session_state.pop("bottomup_tree")
                st.session_state.pop("topdown_tree")
                
            if compare_button or "bottomup_tree" not in st.session_state:
                with st.spinner("Processing 'The cat is fat' with both Tree-LSTM approaches..."):
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    bottomup_tree, topdown_tree = run_example_both_treelstm(device=device, d_in=input_dim, d_hidden=hidden_dim)
                    st.session_state.bottomup_tree = bottomup_tree
                    st.session_state.topdown_tree = topdown_tree
        
        if "bottomup_tree" in st.session_state and "topdown_tree" in st.session_state:
            # Visualization of trees
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Bottom-Up Tree-LSTM Results")
                # Create visualization of bottom-up tree using the tree visualization function
                fig = visualize_tree(st.session_state.bottomup_tree, show_gates=True, title="Bottom-Up Processing")
                st.pyplot(fig)
                
                # Add explanation about what's happening
                st.markdown("""
                **Bottom-Up Flow**: 
                * Leaf nodes initialize with their embeddings
                * Information flows upwards through the tree
                * Parent nodes integrate information from all their children
                * The forget gate determines how much information to keep from each child
                """)
            
            with col2:
                st.markdown("### Top-Down Tree-LSTM Results")
                # Create visualization of top-down tree using the tree visualization function
                fig = visualize_tree(st.session_state.topdown_tree, show_gates=True, title="Top-Down Processing")
                st.pyplot(fig)
                
                # Add explanation about what's happening
                st.markdown("""
                **Top-Down Flow**:
                * Root node initializes with its embedding
                * Information flows downwards through the tree
                * Child nodes receive context from their parent
                * The forget gate determines how much parent information to incorporate
                """)
            
            # Add key observations section
            st.markdown("### Key Observations")
            st.markdown("""
            1. **Information Flow**: Notice how the gate activations differ between the two approaches.
            
            2. **Representation Differences**: 
               * Bottom-up: Nodes contain compositional information from their subtrees
               * Top-down: Nodes contain contextual information from their ancestors
               
            3. **Practical Applications**:
               * Bottom-up: Better for tasks requiring composition (sentiment analysis, classification)
               * Top-down: Better for tasks requiring context distribution (question answering, generation)
               
            4. **Combined Approaches**: Most state-of-the-art models use bidirectional approaches that combine both bottom-up and top-down information flow.
            """)

    # Tab 4: Applications
    with tab4:
        st.header("Applications of Tree-LSTM")
        
        st.markdown("""
        Tree-LSTMs excel in tasks that benefit from modeling hierarchical structure in data:
        """)
        
        application_data = {
            'Application': [
                'Sentiment Analysis', 
                'Semantic Relatedness', 
                'Natural Language Inference',
                'Constituency Parsing',
                'Code Understanding',
                'XML/HTML Processing',
                'Knowledge Graph Reasoning'
            ],
            'Description': [
                'Analyze sentiment by composing meanings hierarchically according to parse structure',
                'Measure semantic similarity between sentences by capturing structural differences',
                'Determine entailment relationships between sentences using tree structures',
                'Build and refine constituency parse trees through recursive processing',
                'Process source code according to its Abstract Syntax Tree (AST) structure',
                'Process structured documents by leveraging their inherent tree structure',
                'Traverse and reason over knowledge graphs by following relationship paths'
            ],
            'Advantage': [
                'Better captures negation and compositional sentiment',
                'More accurate comparison of structurally different sentences',
                'Better understanding of logical relationships that depend on structure',
                'Natural fit for recursive parse tree construction',
                'Respects the syntactic structure of programming languages',
                'Naturally aligns with the hierarchical document structure',
                'Models transitive relationships and complex reasoning paths'
            ]
        }
        
        app_df = pd.DataFrame(application_data)
        st.dataframe(app_df, use_container_width=True, hide_index=True)
        
        # Example case: Sentiment Analysis
        st.subheader("Example Case: Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(r"""
            Consider the sentence: "The movie was not very good"
            
            Parse tree:
            ```
                       S
                      / \
                     /   \
                    NP    VP
                    |    / \
                   The   V  ADJP
                        |  /  \
                       was NEG ADJ
                          |    |
                         not  good
            ```
            
            Standard LSTM may struggle with the negation, but Tree-LSTM can capture it correctly through compositional processing.
            """)
        
        with col2:
            # Create a simple example visualization
            st.markdown("""
            #### Bottom-up Processing in Action
            
            1. Process leaf nodes "The", "was", "not", "good"
            2. Combine "not" + "good" → negative sentiment at ADJP
            3. Combine "was" + ADJP → negative sentiment at VP
            4. Combine "The" + VP → negative sentiment at S (root)
            
            The hierarchical processing correctly captures that "not good" has a negative sentiment, even though "good" alone would be positive.
            """)

    # Tab 5: Resources
    with tab5:
        st.header("Resources & Further Reading")
        
        st.markdown("""
        ### Academic Papers
        
        * [**Improved Semantic Representations From Tree-Structured LSTM Networks**](https://aclanthology.org/P15-1150/) - Kai Sheng Tai, Richard Socher, Christopher D. Manning (ACL 2015)
          * The original Tree-LSTM paper that introduced the architecture
          
        * [**Tree-structured composition in neural networks without tree-structured architectures**](https://aclanthology.org/W15-4007/) - Jacob Andreas, Marcus Rohrbach, Trevor Darrell, Dan Klein (NAACL Workshop 2015)
          * Shows how to approximate tree processing without explicit tree structures
          
        * [**Tree-to-Sequence Attentional Neural Machine Translation**](https://aclanthology.org/P16-1078/) - Akiko Eriguchi, Kazuma Hashimoto, Yoshimasa Tsuruoka (ACL 2016)
          * Uses Tree-LSTM for machine translation by encoding source sentences as trees
        
        ### Tutorials & Implementations
        
        * [PyTorch Tree-LSTM Implementation](https://github.com/jihunchoi/tree-lstm-pytorch) 
          * Clean implementation of various Tree-LSTM architectures
          
        * [DyNet Tree-LSTM Tutorial](https://github.com/neubig/nn4nlp-code/tree/master/04-sequences)
          * Tutorial implementation as part of the Neural Networks for NLP course
          
        * [Tree-LSTM for Sentiment Analysis](https://github.com/stanfordnlp/treelstm)
          * Stanford's original implementation focused on sentiment analysis
        
        ### Relation to Modern Transformer Models
        
        While newer Transformer-based models like BERT, GPT, and T5 have largely superseded Tree-LSTMs for many NLP tasks, the hierarchical processing insights from Tree-LSTMs remain valuable. Recent research explores:
        
        * Incorporating syntactic structure into Transformers
        * Using Tree-LSTMs as specialized components in larger neural architectures
        * Combining the strengths of tree-structured models with the representational power of Transformers
        
        Tree-structured models remain particularly valuable in domains where hierarchical structure is explicitly known and important, such as code analysis, knowledge graphs, and certain types of symbolic reasoning.
        """)

    # Footer
    st.markdown("---")
    st.markdown("Created for educational purposes. Tree-LSTM implementations based on Tai et al. (2015)")

if __name__ == "__main__":
    main() 