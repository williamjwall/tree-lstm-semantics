import streamlit as st
import json
import graphviz
import torch
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import traceback
matplotlib.use('Agg')  # Use non-interactive backend

# Ensure we're using CPU only and avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

try:
    from src.tree_lstm_viz.model import TreeLSTMEncoder
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure to run setup.sh first and activate the virtual environment")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Tree-LSTM Visualizer",
    page_icon="🌳",
    layout="wide"
)

# Initialize encoder
@st.cache_resource
def get_encoder():
    try:
        return TreeLSTMEncoder()
    except Exception as e:
        st.error(f"Failed to initialize TreeLSTMEncoder: {str(e)}")
        return None

# Helper function to generate color gradient based on value
def get_color(val, min_val, max_val):
    # Simple RGB gradient from blue to red
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (val - min_val) / (max_val - min_val)
    
    r = min(255, int(normalized * 255))
    b = min(255, int((1 - normalized) * 255))
    g = min(100, int(100 - abs(normalized - 0.5) * 200))
    
    return f"#{r:02x}{g:02x}{b:02x}"

# Title and description
st.title("Tree-LSTM Semantic Visualizer")
st.markdown("""
This application visualizes constituency parse trees of sentences using:
1. A 3D interactive visualization where node positions reflect semantic representations
2. A traditional top-down 2D tree diagram
3. Additional semantic relationship visualizations

Below are the conceptual explanations for each visualization:

**Note:** Each node includes a 'span' attribute that indicates the range of token indices in the sentence. The semantic values used for coloring (from the third dimension, h[2]) capture key semantic features learned by the model.
""")

with st.expander("Understanding The Visualization", expanded=True):
    st.markdown("""
    ## Key Concepts in Tree-LSTM Visualization

    **Understanding Node Colors and Semantic Values:**
    - **What are the semantic values?**
      - Each node has a hidden state vector (128 dimensions) produced by the Tree-LSTM network
      - These values represent learned semantic features that capture meaning
      - The semantic values used for coloring are taken from the 3rd dimension (h[2]) of the hidden state
      - The first dimensions often encode higher-level semantic properties

    - **Significance of Semantic Dimension 3:**
      - The color of each node is based on the 3rd dimension of its hidden state (h[2])
      - This dimension was selected because it often captures interesting semantic distinctions
      - **Purple/Blue** nodes have lower values in this dimension
      - **Yellow/Green** nodes have higher values

    - **Interpreting these values:**
      - Similar colors indicate nodes with similar semantic properties
      - Color patterns reveal how meaning is composed through the tree
      - When comparing parent nodes to children, color shifts indicate how meaning transforms
      - This dimension may capture features like: concreteness vs. abstractness, entity vs. action, sentiment, or tense information
    
    1. **Node Meaning:**
       - Each node represents a word or phrase from the sentence
       - Leaf nodes are individual words
       - Internal nodes are phrases that combine their children
       - The root node represents the entire sentence
       
    2. **Semantic Relationships:**
       - Nodes that appear close together have similar meanings
       - Distance between nodes represents meaning differences
       - Hovering shows the actual values that make up each node's meaning
       
    3. **Understanding Composition:**
       - Watch how the Tree-LSTM combines smaller meanings into larger ones
       - Parent nodes represent abstractions of their children
       - The size of nodes shows their "semantic richness"
       
    4. **Compare Different Views:**
       - Try all three visualization modes with the same sentence
       - Notice how different aspects of meaning become visible in each mode
       - Use the tree view to understand structure, and 3D view to see meaning
       
    5. **Understanding Colors:**
       - The consistent color scheme shows semantic dimension 3
       - The same color in different visualizations represents the same semantic value
       - Track how this semantic aspect flows from lower to higher levels in the tree
    6. **Understanding the 'span' Attribute:**
       - The 'span' attribute defines the start and end token indices of the original sentence that form the node. This helps correlate each node with its position in the sentence.
    """)

# Conceptual explanations in header with collapsible sections
with st.expander("Tree Structure + Semantic Dimensions (Visualization Guide)", expanded=False):
    st.markdown("""
    ## Tree Structure + Semantic Dimensions
    
    **What you're seeing:**
    This visualization shows how meaning is structured within the grammar of a sentence. 
    
    **How it works:**
    - The **X and Y positions** come directly from the first two dimensions of each node's semantic meaning
    - The **Z position** (vertical) shows the grammatical depth in the tree
    - **Bigger nodes** have more "semantic weight" or importance
    - **Node colors** represent another aspect of meaning (the third semantic dimension)
    
    **Why it's useful:**
    This approach lets you see how meaning is built up through the grammatical structure. Words and phrases 
    that have similar meanings will appear closer together on the X-Y plane, while their grammatical 
    relationship is preserved vertically.
    
    **What to look for:**
    - Similar words/phrases clustered together horizontally
    - How meaning flows and transforms as you move up the tree
    - Parent nodes that combine and abstract the meaning of their children
    """)

with st.expander("Pure Semantic Space (PCA) (Visualization Guide)", expanded=False):
    st.markdown("""
    ## Pure Semantic Space

    **What you're seeing:**
    This visualization shows the pure meaning relationships between words and phrases, freed from grammatical constraints.
    
    **How it works:**
    - All dimensions (X, Y, and Z) represent semantic meaning rather than grammar
    - Similar meanings cluster together in 3D space
    - Tree connections are preserved but don't constrain where nodes appear
    - PCA (Principal Component Analysis) finds the most important patterns in the meaning vectors
    
    **Why it's useful:**
    This approach reveals semantic relationships that might be hidden by the tree structure. It's like seeing the 
    "meaning landscape" of the sentence, where similar concepts naturally group together.
    
    **What to look for:**
    - Clusters of semantically related words and phrases
    - The root node's position relative to its parts
    - How different types of phrases (noun phrases, verb phrases) form distinct regions
    - Outliers that carry unique meaning in the sentence
    """)

with st.expander("Hybrid View (Visualization Guide)", expanded=False):
    st.markdown("""
    ## Hybrid View
    
    **What you're seeing:**
    This visualization balances both grammar and meaning in one view.
    
    **How it works:**
    - The **X and Y positions** show semantic similarity between words/phrases
    - The **Z position** (vertical) preserves the grammatical tree structure
    - Nodes at the same height belong to the same grammatical level
    - Similar meanings cluster together horizontally at each level
    
    **Why it's useful:**
    This approach gives you the best of both worlds - you can see how meaning relates to structure.
    It shows which parts of speech have similar meanings while maintaining the grammatical hierarchy.
    
    **What to look for:**
    - How meaning clusters horizontally at each grammatical level
    - The progression of meaning as you move up the tree
    - Comparisons between phrases at the same grammatical depth
    - The relationship between a parent's meaning and its children's meanings
    """)

with st.expander("Semantic Similarity Heatmap (Visualization Guide)", expanded=False):
    st.markdown("""
    ## Semantic Similarity Heatmap
    
    **What you're seeing:**
    This visualization shows how similar or different the meanings of various sentence constituents are to each other.
    
    **How it works:**
    - Each cell shows the semantic similarity between two constituents
    - Brighter/warmer colors indicate higher similarity
    - The diagonal is always brightest (self-similarity = 1.0)
    - The root node is included for comparison with all other constituents
    
    **Why it's useful:**
    This approach reveals which parts of the sentence are semantically related, even if they're 
    not close in the tree structure. It helps you identify which phrases contribute most to the 
    overall sentence meaning.
    
    **What to look for:**
    - Which phrases are most similar to the root (overall meaning)
    - Clusters of mutually similar constituents
    - Phrases that have little similarity with others (semantic outliers)
    - How similarity patterns relate to grammatical roles
    """)
    
with st.expander("Phrasal Structure Diagram (Visualization Guide)", expanded=False):
    st.markdown("""
    ## Phrasal Structure Diagram
    
    **What you're seeing:**
    This visualization shows the hierarchical structure of phrases with semantic information encoded in color.
    
    **How it works:**
    - Concentric rings represent tree depth levels
    - Each segment represents a constituent/phrase
    - Size reflects the phrase's structural importance
    - Color shows semantic value (using the same dimension as the 3D view)
    
    **Why it's useful:**
    This visualization makes it easy to see how phrases nest within each other and how 
    semantic properties propagate up through the structure. It provides a compact overview 
    of the entire sentence structure.
    
    **What to look for:**
    - How colors (semantic values) change as you move from inner to outer rings
    - The distribution of semantic values across different phrase types
    - How the root's semantic value relates to its constituent parts
    - Patterns in how meaning is composed hierarchically
    
    **Note on colors:**
    - Yellow/green shades indicate higher values in semantic dimension 3
    - Purple/blue shades indicate lower values
    - This allows you to track one aspect of meaning throughout the tree structure
    """)

# Device info
st.sidebar.info("Running on: CPU")

# Input with example
example = "Julia kindly gave milk to a very friendly new neighbor after going to the river bank"
st.markdown("<h3 style='font-weight: bold;'>Enter a sentence:</h3>", unsafe_allow_html=True)
sentence = st.text_input(
    "",
    value=example,
    key="sentence_input",
    help="The sentence will be parsed into a constituency tree"
)

# Get encoder
encoder = get_encoder()
if encoder is None:
    st.error("Failed to initialize encoder. Please check the console for errors.")
    st.stop()

if sentence:
    try:
        # Process sentence
        with st.spinner("Building parse tree..."):
            result = encoder.encode(sentence)
        
        # Create tabs for different visualizations - REORDERED to put 3D first
        tab1, tab2, tab3, tab4 = st.tabs([
            "3D Tree Visualization",
            "2D Tree Visualization", 
            "Semantic Similarity Heatmap",
            "Phrasal Structure Diagram"
        ])
        
        # Extract all hidden states from the tree for use in all visualizations
        all_vectors = []
        node_info = []
        node_labels = []
        
        def collect_vectors(tree, level=0, is_leaf=False, parent_idx=-1):
            if tree['h'] is not None:
                h = np.array(tree['h'])
                all_vectors.append(h)
                
                # Create a descriptive label for the node
                span = tree['span']
                if is_leaf and span[0] < len(sentence.split()):
                    token_text = sentence.split()[span[0]] if span[0] == span[1] - 1 else "..."
                    label = f"{tree['label']}: {token_text}"
                else:
                    # For non-leaf nodes, include the covered text
                    if span[0] < len(sentence.split()) and span[1] <= len(sentence.split()):
                        covered_text = " ".join(sentence.split()[span[0]:span[1]])
                        if len(covered_text) > 20:
                            covered_text = covered_text[:17] + "..."
                        label = f"{tree['label']}: {covered_text}"
                    else:
                        label = tree['label']
                
                node_labels.append(label)
                node_info.append({
                    'tree': tree, 
                    'level': level, 
                    'is_leaf': is_leaf,
                    'parent_idx': parent_idx,
                    'label': label
                })
                idx = len(all_vectors) - 1
            else:
                # Fallback for nodes without vectors
                idx = -1
            
            # Process children
            for child in tree.get('children', []):
                collect_vectors(child, level + 1, len(child.get('children', [])) == 0, idx)
            
            return idx
        
        # Collect vectors
        root_idx = collect_vectors(result['tree'])
        
        # Convert to array for processing
        if all_vectors:
            vectors_array = np.array(all_vectors)
            
            # 3D Visualization - Now in tab1 (first tab)
            with tab1:
                st.subheader("3D Parse Tree with Semantic Layout")
                
                # Enhanced explanation
                st.markdown("""
                This visualization shows the parse tree in three dimensions, with semantic information determining node positions.
                
                **Node Properties:**
                - **Position (X, Y)**: Reflects semantic dimensions from the hidden state
                - **Position (Z)**: Reflects tree structure/depth
                - **Size**: Proportional to semantic vector magnitude
                - **Color**: Based on the third semantic dimension (h[2])
                - **Hover text**: Shows node label and first few hidden state dimensions
                """)
                
                # Visualization options
                viz_mode = st.radio(
                    "Visualization Mode:",
                    ["Tree Structure + Semantic Dimensions", "Pure Semantic Space (PCA)", "Hybrid View"],
                    index=0,
                    help="Choose how to visualize the tree in 3D space"
                )
                
                # Apply PCA for semantic dimensions if needed
                if viz_mode in ["Pure Semantic Space (PCA)", "Hybrid View"]:
                    pca = PCA(n_components=3)
                    semantic_coords = pca.fit_transform(vectors_array)
                
                # Generate 3D coordinates for each node
                x, y, z = [], [], []
                colors = []
                sizes = []
                labels = []
                
                # Create edges
                x_edges, y_edges, z_edges = [], [], []
                
                # Scale factor for semantic dimensions
                scale = 10.0
                
                # Process each node to determine its 3D position
                for i, (vec, info) in enumerate(zip(all_vectors, node_info)):
                    tree = info['tree']
                    level = info['level']
                    is_leaf = info['is_leaf']
                    parent_idx = info['parent_idx']
                    
                    # Determine position based on visualization mode
                    if viz_mode == "Tree Structure + Semantic Dimensions":
                        # Use tree structure for z, and semantic values for x, y
                        # Scale semantic values to reasonable range
                        pos_x = vec[0] * scale
                        pos_y = vec[1] * scale
                        pos_z = -level * 3  # Negative to make root at top
                        
                    elif viz_mode == "Pure Semantic Space (PCA)":
                        # Use PCA-reduced dimensions
                        pos_x = semantic_coords[i, 0] * scale
                        pos_y = semantic_coords[i, 1] * scale
                        pos_z = semantic_coords[i, 2] * scale
                        
                    else:  # Hybrid View
                        # Use tree level for z but semantic values for x, y
                        pos_x = semantic_coords[i, 0] * scale
                        pos_y = semantic_coords[i, 1] * scale
                        pos_z = -level * 2  # Preserve some tree structure
                    
                    # Add coordinates
                    x.append(pos_x)
                    y.append(pos_y)
                    z.append(pos_z)
                    
                    labels.append(info['label'])
                    
                    # Use vector magnitude for node size (normalized)
                    vec_mag = np.linalg.norm(vec)
                    size = 10 + vec_mag / 3
                    sizes.append(min(20, size))  # Cap size
                    
                    # Determine color based on 3rd semantic dimension
                    # Use a consistent dimension for coloring
                    color_val = vec[2]
                    colors.append(color_val)
                    
                    # Add edge from parent node if exists
                    if parent_idx >= 0:
                        x_edges.extend([x[parent_idx], pos_x, None])
                        y_edges.extend([y[parent_idx], pos_y, None])
                        z_edges.extend([z[parent_idx], pos_z, None])
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add edges (lines)
                fig.add_trace(go.Scatter3d(
                    x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode='lines',
                    line=dict(color='rgba(50,50,50,0.8)', width=2),
                    hoverinfo='none',
                    name='Tree Connections'
                ))
                
                # Normalize colors for consistent colorscale
                if colors:
                    min_val = min(colors)
                    max_val = max(colors)
                    if min_val == max_val:
                        normalized_colors = [0.5] * len(colors)
                    else:
                        normalized_colors = [(c - min_val) / (max_val - min_val) for c in colors]
                else:
                    normalized_colors = []
                
                # Add nodes
                fig.add_trace(go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',  # Remove text mode to avoid null DOM element issues
                    marker=dict(
                        size=sizes,
                        color=normalized_colors,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title="Semantic Dimension 3")
                    ),
                    # text= removed to avoid DOM issues
                    hovertext=[f"{l}<br>Hidden[0:3]=[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]<br>Vector magnitude: {np.linalg.norm(v):.2f}" 
                              for l, v in zip(labels, all_vectors)],
                    hoverinfo='text',
                    name='Tree Nodes'
                ))
                
                # Update layout
                axis_title_font = dict(size=14)
                
                # Different axis titles based on visualization mode
                if viz_mode == "Tree Structure + Semantic Dimensions":
                    x_title = "Semantic Dimension 1"
                    y_title = "Semantic Dimension 2"
                    z_title = "Tree Level (Syntax)"
                elif viz_mode == "Pure Semantic Space (PCA)": 
                    x_title = "PCA Dimension 1"
                    y_title = "PCA Dimension 2"
                    z_title = "PCA Dimension 3"
                else:  # Hybrid view
                    x_title = "Semantic PCA Dimension 1"
                    y_title = "Semantic PCA Dimension 2"
                    z_title = "Tree Level (Syntax)"
                
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title=x_title, title_font=axis_title_font),
                        yaxis=dict(title=y_title, title_font=axis_title_font),
                        zaxis=dict(title=z_title, title_font=axis_title_font)
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene_camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=-1.5, z=0.5)
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Add Plotly config to fix web worker issues
                config = {
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'tree_lstm_viz'
                    },
                    'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'lasso2d'],
                    'responsive': True,
                    # Disable web workers to avoid errors
                    'showLink': False,
                    'linkText': '',
                    'plotlyServerURL': ''
                }
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True, config=config)
                
            
            # 2D Tree Visualization - Now in tab2 (second tab)
            with tab2:
                st.subheader("Constituency Parse Tree")
                
                st.markdown("""
                This is a traditional top-down tree visualization showing the grammatical structure of the sentence.
                Each node shows:
                - The grammatical category (NP, VP, S, etc.)
                - The span of tokens it covers in the sentence
                - For leaf nodes, the actual token text
                """)
                
                # Create Graphviz visualization
                dot = graphviz.Digraph()
                dot.attr(rankdir='TB')
                
                # Process tree nodes
                def add_node_to_graph(tree, parent_id=None):
                    node_id = str(id(tree))
                    
                    # Format label
                    label = f"{tree['label']} {tree['span']}"
                    
                    # Determine node style
                    if 'children' in tree and tree['children']:
                        # Internal node
                        dot.node(node_id, label, shape="box", style="rounded")
                    else:
                        # Leaf node (token)
                        span = tree['span']
                        if span[0] == span[1] - 1:  # Single token
                            label += f"\n→ '{sentence.split()[span[0]]}'" if span[0] < len(sentence.split()) else ""
                        dot.node(node_id, label)
                    
                    # Connect to parent
                    if parent_id is not None:
                        dot.edge(parent_id, node_id)
                    
                    # Process children recursively
                    for child in tree.get('children', []):
                        add_node_to_graph(child, node_id)
                
                # Build the graph
                add_node_to_graph(result['tree'])
                st.graphviz_chart(dot, use_container_width=True)
                
                # Add explanation of node labels
                st.markdown("""
                **Node Label Guide:**
                - **S**: Sentence
                - **NP**: Noun Phrase
                - **VP**: Verb Phrase
                - **PP**: Prepositional Phrase
                - **ADJP**: Adjective Phrase
                - **ADVP**: Adverb Phrase
                - **Numbers in parentheses**: Token span indices

                **Note:** The span attribute indicates the range of token indices in the sentence that the node covers.
                """)
            
            with tab3:
                st.subheader("Semantic Similarity Heatmap")
                
                st.markdown("""
                This heatmap shows how semantically similar each constituent is to every other constituent in the sentence.
                Bright/yellow cells indicate high similarity, while dark/purple cells indicate low similarity.
                """)
                
                # Compute semantic similarity matrix
                sim_matrix = cosine_similarity(vectors_array)
                
                # Create a heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=sim_matrix,
                    x=node_labels,
                    y=node_labels,
                    colorscale='Viridis',
                    colorbar=dict(title="Cosine Similarity"),
                    hovertemplate='Similarity between %{y} and %{x}: %{z:.3f}<extra></extra>'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Semantic Similarity Between Constituents",
                    height=600,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis=dict(tickangle=-45),
                )
                
                # Show the plot with config options
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True,
                    'toImageButtonOptions': {
                        'format': 'png', 
                        'filename': 'similarity_heatmap'
                    }
                })
                
                # Add explanation about root similarity
                if len(node_labels) > 0:
                    st.subheader("Root Similarity Analysis")
                    
                    # Add explanation
                    st.markdown("""
                    The chart below shows which phrases in the sentence are most semantically similar to the root (entire sentence meaning).
                    Higher bars indicate constituents that carry more of the overall sentence meaning.
                    """)
                    
                    # Get similarities to root
                    root_similarities = sim_matrix[0]  # Assuming root is the first node
                    
                    # Create a dataframe for display
                    sim_data = {
                        'Constituent': node_labels,
                        'Similarity to Root': root_similarities
                    }
                    
                    # Sort by similarity
                    sorted_indices = np.argsort(-root_similarities)  # Descending
                    sorted_constituents = [node_labels[i] for i in sorted_indices]
                    sorted_similarities = [root_similarities[i] for i in sorted_indices]
                    
                    # Create bar chart
                    fig = go.Figure(data=go.Bar(
                        x=sorted_constituents[:10],  # Top 10 for readability
                        y=sorted_similarities[:10],
                        marker=dict(
                            color=sorted_similarities[:10],
                            colorscale='Viridis',
                            colorbar=dict(title="Similarity")
                        ),
                        text=[f"{sim:.3f}" for sim in sorted_similarities[:10]],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Top Constituents by Similarity to Root",
                        xaxis_title="Constituent",
                        yaxis_title="Semantic Similarity",
                        height=400
                    )
                    
                    # Show plot with simplified config
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                    })
                    
                    st.markdown("""
                    **Interpreting This Chart:**
                    - A similarity of 1.0 means identical meaning to the root
                    - Higher values indicate phrases that capture more of the overall sentence meaning
                    - Main phrases/clauses typically have high similarity to the root
                    - Individual words or small phrases usually have lower similarity
                    """)
            
            # Use try-except block for the Phrasal Structure Diagram
            with tab4:
                st.subheader("Phrasal Structure Diagram")
                
                # Extra explanation
                st.markdown("""
                This visualization shows the hierarchical structure of the parse tree in a simplified format.
                Colors represent the same semantic dimension (3) used in the 3D visualization.
                """)
                
                # Create a simpler tree representation
                with st.spinner("Building hierarchical tree data..."):
                    # First create a flattened representation with level info
                    flat_tree_data = []
                    
                    def traverse_tree(node, level=0, parent_label="Root"):
                        if node.get('h') is None:
                            return
                            
                        # Get the node's text span
                        span = node['span']
                        if span[0] < len(sentence.split()) and span[1] <= len(sentence.split()):
                            covered_text = " ".join(sentence.split()[span[0]:span[1]])
                            if len(covered_text) > 30:
                                covered_text = covered_text[:27] + "..."
                        else:
                            covered_text = ""
                        
                        # Get semantic value for coloring
                        sem_value = node['h'][2] if node['h'] is not None else 0
                        
                        # Store node data
                        flat_tree_data.append({
                            "level": level,
                            "label": node['label'],
                            "text": covered_text,
                            "parent": parent_label,
                            "semantic_value": sem_value
                        })
                        
                        # Process children
                        for child in node.get('children', []):
                            traverse_tree(child, level + 1, node['label'])
                    
                    # Start traversal from root
                    traverse_tree(result['tree'])
                
                # Visualize the tree structure as a table with indentation
                if flat_tree_data:
                    # Find min and max semantic values for color normalization
                    sem_values = [node["semantic_value"] for node in flat_tree_data]
                    min_val = min(sem_values)
                    max_val = max(sem_values) if max_val != min_val else min_val + 1
                    
                    # Create a formatted display table
                    st.markdown("### Hierarchical Tree Structure")
                    
                    # Import pandas for better dataframe display
                    import pandas as pd
                    
                    # Function to get color from viridis-like palette
                    def get_color_hex(value):
                        # Normalize the value
                        if max_val == min_val:
                            norm_val = 0.5
                        else:
                            norm_val = (value - min_val) / (max_val - min_val)
                        
                        # Simple approximation of viridis color scale
                        if norm_val < 0.25:
                            return "#440154"  # Deep purple
                        elif norm_val < 0.5:
                            return "#3b528b"  # Blue
                        elif norm_val < 0.75:
                            return "#21918c"  # Teal
                        else:
                            return "#fde725"  # Yellow
                    
                    # Create a new column that shows the indented label
                    for node in flat_tree_data:
                        level = node["level"]
                        label = node["label"]
                        # Create indentation
                        indent = "  " * level
                        node["tree_display"] = f"{indent}{'└─ ' if level > 0 else ''}{label}"
                    
                    # Create a DataFrame
                    df = pd.DataFrame(flat_tree_data)
                    
                    # Select and rename columns for display
                    display_df = df[["tree_display", "text", "semantic_value"]].copy()
                    display_df.columns = ["Tree Structure", "Text Span", "Semantic Value"]
                    
                    # Show the DataFrame
                    st.dataframe(
                        display_df,
                        column_config={
                            "Semantic Value": st.column_config.NumberColumn(
                                "Semantic Value",
                                format="%.3f",
                                help="Value of semantic dimension 3"
                            ),
                        },
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add explanation of semantic dimension 3
                    st.markdown("""
                    **Understanding Semantic Values:**
                    
                    The "Semantic Value" shown above is the value of dimension 3 in each node's hidden state vector.
                    This value represents a specific semantic feature that the Tree-LSTM has learned, which may relate to:
                    
                    - The grammatical role of the phrase
                    - The semantic category (entity, action, property)
                    - The level of abstraction or specificity
                    """)
                    
                    # Also show a more visually compelling tree representation
                    st.markdown("### Visual Tree Structure")
                    
                    # Create a more compact visual representation with colored boxes
                    cols = st.columns(10)
                    
                    # Calculate levels for display
                    levels = sorted(list(set(node["level"] for node in flat_tree_data)))
                    
                    # Group nodes by level
                    nodes_by_level = {}
                    for node in flat_tree_data:
                        level = node["level"]
                        if level not in nodes_by_level:
                            nodes_by_level[level] = []
                        nodes_by_level[level].append(node)
                    
                    # Display colored boxes for each level
                    for level in levels:
                        st.markdown(f"**Level {level}:**")
                        
                        # Create columns for this level
                        level_cols = st.columns(min(8, len(nodes_by_level[level])))
                        
                        # Display nodes
                        for i, node in enumerate(nodes_by_level[level]):
                            col_idx = i % len(level_cols)
                            label = node["label"]
                            text = node["text"]
                            sem_val = node["semantic_value"]
                            color = get_color_hex(sem_val)
                            
                            # Create a colored box with node info
                            level_cols[col_idx].markdown(
                                f"""
                                <div style="background-color: {color}; color: white; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
                                <strong>{label}</strong><br/>
                                <small>{text[:20] + '...' if len(text) > 20 else text}</small><br/>
                                <small>Value: {sem_val:.3f}</small>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    
                    # Also create a simple bar chart showing semantic values by level
                    st.markdown("### Semantic Values by Tree Depth")
                    
                    # Group data by level
                    level_groups = {}
                    for node in flat_tree_data:
                        level = node["level"]
                        if level not in level_groups:
                            level_groups[level] = []
                        level_groups[level].append(node["semantic_value"])
                    
                    # Calculate average semantic value by level
                    level_avgs = {level: sum(values)/len(values) for level, values in level_groups.items()}
                    
                    # Create a simple bar chart
                    levels = sorted(level_groups.keys())
                    avg_values = [level_avgs[level] for level in levels]
                    
                    # Use Plotly for the bar chart
                    fig = go.Figure(data=go.Bar(
                        x=[f"Level {level}" for level in levels],
                        y=avg_values,
                        marker=dict(
                            color=avg_values,
                            colorscale='Viridis',
                            colorbar=dict(title="Avg. Semantic Value")
                        ),
                        text=[f"{val:.3f}" for val in avg_values],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Average Semantic Value by Tree Depth",
                        xaxis_title="Tree Level",
                        yaxis_title="Average Semantic Value",
                        height=400
                    )
                    
                    # Show plot with minimal config
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': False,
                        'staticPlot': True  # Make it a static plot to avoid issues
                    })
                    
                    st.markdown("""
                    **Interpreting the Chart:**
                    - This chart shows how semantic values in dimension 3 change as you move deeper into the tree
                    - Higher bars indicate levels with higher average semantic values
                    - This pattern reveals how this specific semantic feature is distributed across syntactic levels
                    - Typically, we might expect certain syntactic levels to consistently encode particular semantic features
                    """)
                
                else:
                    st.warning("No hierarchical data available for visualization.")
        
        else:
            st.error("No semantic vectors available in the tree")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try a different sentence or check the console for details.") 