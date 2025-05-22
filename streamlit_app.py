import streamlit as st
import json
import graphviz
import torch
import os
import sys
import warnings
import time
import traceback

# Silence all warnings including from transformers
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Set NLTK_DATA environment variable
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

# Set page config first
st.set_page_config(
    page_title="Tree-LSTM Visualizer",
    page_icon="src/visualization/sakura.png",
    layout="wide"
)

# Title and description - moved to before loading container
st.title("Tree-LSTM Semantic Visualizer")

# Create application initialization state
init_complete = False

# Define initialization steps for accurate progress tracking
init_steps = {
    "dependencies": {"weight": 0.2, "completed": False, "message": "Loading dependencies..."},
    "numpy": {"weight": 0.1, "completed": False, "message": "Loading numerical libraries..."},
    "visualization": {"weight": 0.1, "completed": False, "message": "Initializing visualization components..."},
    "ml_libs": {"weight": 0.1, "completed": False, "message": "Loading machine learning libraries..."},
    "nlp_pipeline": {"weight": 0.2, "completed": False, "message": "Configuring NLP pipeline..."},
    "benepar": {"weight": 0.1, "completed": False, "message": "Setting up parsing components..."},
    "encoder": {"weight": 0.2, "completed": False, "message": "Loading semantic encoder..."}
}

# Function to calculate current progress based on completed steps
def calculate_progress():
    completed_weight = sum(step["weight"] for step in init_steps.values() if step["completed"])
    return completed_weight

# Show loading screen while initialization is in progress
loading_container = st.container()
with loading_container:
    # Custom CSS for a more professional, thicker loading bar
    st.markdown("""
    <style>
    .stProgress {
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .stProgress > div > div > div > div {
        background-color: #4a86e8;
        height: 12px;
        border-radius: 4px;
    }
    .stProgress > div > div > div {
        background-color: rgba(74, 134, 232, 0.15);
        height: 12px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Better layout with more space for the loading bar
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    loading_col1, loading_col2, loading_col3 = st.columns([1, 3, 1])
    with loading_col2:
        # Center the image with CSS
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("<div style='text-align: center; color: #4a86e8; font-size: 16px; margin-top: 10px; font-weight: 500;'>Loading dependencies...</div>", unsafe_allow_html=True)

        # Function to update loading progress
        def update_progress(step_key, completed=True):
            if step_key in init_steps:
                init_steps[step_key]["completed"] = completed
                progress_value = calculate_progress()
                message = init_steps[step_key]["message"]
                
                # Update the progress bar and status message
                progress_bar.progress(progress_value)
                status_text.markdown(f"<div style='text-align: center; color: #4a86e8; font-size: 16px; margin-top: 10px; font-weight: 500;'>{message}</div>", unsafe_allow_html=True)
                
                # Small delay to make progress visible
                time.sleep(0.1)

# Start tracking dependencies
update_progress("dependencies")

# Check for numpy compatibility - simplified
try:
    import numpy as np
    # No need to show NumPy version in sidebar
    update_progress("numpy")
except ImportError as e:
    st.error(f"Error importing NumPy: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"NumPy compatibility issue: {str(e)}")
    st.warning("Trying to fix NumPy compatibility...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.3"])
        import numpy as np
        update_progress("numpy")
    except Exception as e2:
        st.error(f"Failed to fix NumPy: {str(e2)}")
        st.stop()

# Now import other dependencies that depend on numpy - simplified logging
try:
    update_progress("visualization")
    from matplotlib import pyplot as plt
    import matplotlib
    # Remove matplotlib version display
    
    import plotly.graph_objects as go
    import plotly.express as px
    
    try:
        from sklearn import __version__ as sklearn_version
        # Remove scikit-learn version display
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity
        update_progress("ml_libs")
    except Exception as e:
        st.error(f"Error importing scikit-learn: {str(e)}")
        st.info("Trying to reinstall scikit-learn...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "scikit-learn==1.3.2"])
            from sklearn import __version__ as sklearn_version
            from sklearn.decomposition import PCA
            from sklearn.metrics.pairwise import cosine_similarity
            update_progress("ml_libs")
        except Exception as e2:
            st.error(f"Failed to fix scikit-learn: {str(e2)}")
    
    import traceback
    import pandas as pd
except ImportError as e:
    st.error(f"Error importing dependencies: {str(e)}")
    st.info("Try reloading the page or contact support.")
    st.stop()

# Patch for transformers warnings - must be done before any transformers import
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass  # transformers not installed

matplotlib.use('Agg')  # Use non-interactive backend

# Simple function to ensure models are loaded with simplified logging
@st.cache_resource
def load_nlp_pipeline():
    try:
        import spacy
        # Simplified version info - moved to a single line in sidebar later
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            
            # Add benepar component if available with less verbosity
            try:
                # First try to use our new BeneparHelper if available
                if structured_logging:
                    helper = BeneparHelper('benepar_en3')
                    nlp = helper.setup_spacy_pipeline(nlp)
                else:
                    # Fall back to old implementation
                    if "benepar" not in nlp.pipe_names:
                        try:
                            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                        except Exception:
                            # Reduced verbosity - silent error handling
                            try:
                                import benepar
                                benepar.download('benepar_en3')
                                nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                            except:
                                pass
            except:
                # Reduced verbosity - silent handling
                pass
            
            return nlp
        except Exception:
            st.error("Error loading spaCy model")
            return None
    except Exception:
        st.error("Error setting up NLP pipeline")
        return None

# Add src directory to path if needed
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to import our new logger and BeneparHelper
try:
    from src.tree_lstm_viz.logger import setup_logger, log_dependency_status, log_model_status
    from src.tree_lstm_viz.benepar_utils import BeneparHelper
    
    # Set up logger
    logger = setup_logger('streamlit_app')
    structured_logging = True
    logger.info("Starting Tree-LSTM Visualizer")
except ImportError:
    structured_logging = False

# Try to download benepar model if needed - simplified logging
try:
    # First try to use our new BeneparHelper
    if structured_logging:
        logger.info("Setting up Benepar using BeneparHelper")
        helper = BeneparHelper('benepar_en3')
        benepar_installed = helper.ensure_benepar_installed()
        
        # Check if model is downloaded
        model_downloaded, model_path = helper.is_model_downloaded()
        if not model_downloaded:
            st.sidebar.info("Downloading benepar model...")
            helper.download_model()
        update_progress("benepar")
    else:
        # Fall back to old implementation
        import nltk
        import benepar
        
        # Check if model exists and download if needed
        model_exists = False
        try:
            import benepar.download as benepar_download
            for path in benepar_download._get_download_dir():
                model_path = os.path.join(path, "benepar_en3")
                if os.path.exists(model_path):
                    model_exists = True
                    break
        except Exception as e:
            pass
        
        if not model_exists:
            st.sidebar.info("Downloading benepar model...")
            try:
                benepar.download('benepar_en3')
            except Exception as e:
                st.sidebar.error(f"Error downloading benepar model: {str(e)}")
        update_progress("benepar")
except Exception as e:
    st.sidebar.warning(f"Benepar not available: {str(e)}")

# Update progress for NLP pipeline
update_progress("nlp_pipeline")

# Initialize NLP pipeline with simplified log output
with st.spinner("Setting up NLP models..."):
    nlp = load_nlp_pipeline()
    if not nlp:
        st.sidebar.error("NLP pipeline initialization failed")

# Initialize encoder
# Add option for processing direction outside the cached function
direction = st.sidebar.radio(
    "TreeLSTM Processing Direction:",
    ["Bottom-Up (Default)", "Top-Down", "Bidirectional"],
    index=0
)

# Map selection to model mode
mode_map = {
    "Bottom-Up (Default)": "bottom-up",
    "Top-Down": "top-down",
    "Bidirectional": "both"
}
mode = mode_map[direction]

@st.cache_resource
def get_encoder(mode="bottom-up"):
    try:
        # Try bidirectional model first
        try:
            from src.tree_lstm_viz.model_bidirectional import BidirectionalTreeLSTMEncoder
            
            # Now try to initialize the encoder with selected mode
            encoder = BidirectionalTreeLSTMEncoder(mode=mode)
            
            # Test the encoder with a simple sentence to verify it works
            test_result = encoder.encode("This is a test.")
            if test_result:
                return encoder, "bidirectional", None
        except ImportError as e:
            # Fall back to standard model
            try:
                from src.tree_lstm_viz.model import TreeLSTMEncoder
                
                # Now try to initialize the encoder (which will handle Benepar setup internally)
                encoder = TreeLSTMEncoder()
                
                # Test the encoder with a simple sentence to verify it works
                test_result = encoder.encode("This is a test.")
                if test_result:
                    return encoder, "standard", None
            except ImportError as e:
                # Try alternative model
                try:
                    from src.tree_lstm_viz.model_alt import TreeLSTMEncoder
                    try:
                        encoder = TreeLSTMEncoder()
                        # Test the encoder with a simple sentence
                        test_result = encoder.encode("This is a test.")
                        return encoder, "alternative", None
                    except Exception as e:
                        return None, None, f"Error initializing alternative TreeLSTMEncoder: {str(e)}"
                except ImportError:
                    return None, None, "Could not import any TreeLSTMEncoder"
            except Exception as e:
                return None, None, f"Error initializing TreeLSTMEncoder: {str(e)}"
        except Exception as e:
            return None, None, f"Error initializing BidirectionalTreeLSTMEncoder: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error in get_encoder: {str(e)}\n{traceback.format_exc()}"

# Update progress for encoder loading
update_progress("encoder", False)  # Mark as in progress

# Initialize encoder with simplified log output
with st.spinner("Loading Tree-LSTM encoder..."):
    result = get_encoder(mode)
    
    if isinstance(result, tuple):
        encoder, model_type, error = result
        
        if encoder:
            # Convert mode to display-friendly format
            display_mode = {"bottom-up": "Bottom-Up", "top-down": "Top-Down", "both": "Bidirectional"}
            if model_type == "bidirectional":
                st.sidebar.success(f"Using {display_mode.get(mode, mode)} TreeLSTM")
            elif model_type == "standard":
                st.sidebar.success("Using Bottom-Up TreeLSTM (Legacy)")
            elif model_type == "alternative":
                st.sidebar.success("Using Alternative TreeLSTM")
                
            update_progress("encoder", True)  # Mark as complete
        else:
            if error:
                st.error(error)
            st.sidebar.error("Tree-LSTM encoder initialization failed")
            update_progress("encoder", False)  # Mark as incomplete
    else:
        encoder = result
        if encoder:
            st.sidebar.success("Tree-LSTM encoder ready")
            update_progress("encoder", True)  # Mark as complete
        else:
            st.sidebar.error("Tree-LSTM encoder initialization failed")
            update_progress("encoder", False)  # Mark as incomplete

# Simplified hardware info display - extremely concise
if encoder:
    import torch
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"Running on: {device}")

# Check if encoder loaded successfully - simplify error message
if encoder is None:
    st.error("Failed to initialize the Tree-LSTM model. Try refreshing the page.")

# Mark initialization as complete
init_complete = True

# Final progress update - ensure we reach 100%
for step in init_steps:
    if not init_steps[step]["completed"]:
        init_steps[step]["completed"] = True

# Show 100% completion
progress_bar.progress(1.0)
status_text.markdown("<div style='text-align: center; color: #4a86e8; font-size: 16px; margin-top: 10px; font-weight: 500;'>Initialization complete</div>", unsafe_allow_html=True)

# Small delay before hiding loading screen
time.sleep(0.5)

# Hide the loading screen once initialization is complete
loading_container.empty()
    
# No welcome message - completely removed

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

# Add resource constraints for Streamlit Cloud
MAX_SENTENCE_LENGTH = 60
MAX_VECTOR_DISPLAY = 20

# Main application content
if init_complete:  # Only show content when initialization is complete

    input_container = st.container()
    with input_container:
        st.markdown('<div class="sentence-input">', unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight: bold;'>Enter a sentence:</h3>", unsafe_allow_html=True)
        
        example = "Julia kindly gave milk to a very friendly new neighbor after going to the river bank"
        sentence = st.text_input(
            "Sentence",
            value=example,
            key="sentence_input",
            help="The sentence will be parsed into a constituency tree",
            label_visibility="collapsed"
        )
        
        # Add examples as bullet points instead of buttons
        st.markdown("**Examples:**")
        st.markdown("""
        * Colorless green ideas sleep furiously.
        * The man who hunts ducks out on weekends.
        * Who did you say that Mary claimed that John believed that Peter saw?
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Documentation - ALL SET TO COLLAPSED (expanded=False)
    with st.expander("Understanding The Visualization", expanded=False):
        st.markdown("""
        ## Key Concepts in Tree-LSTM Visualization

        ### Semantic Representation
        Every node in the tree contains a **hidden state vector** (a multi-dimensional representation of meaning) produced by the Tree-LSTM network. 
        These vectors encode semantic properties that capture various aspects of meaning - from simple features like tense or number to complex 
        semantic relationships.

        ### Visual Elements and Their Meaning
        
        | Element | What It Represents |
        |---------|-------------------|
        | **Node Position** | In 3D space, positions reflect semantic dimensions - similar meanings appear closer together. For example, words like "river" and "bank" might cluster near each other if they share semantic properties, even if they're far apart in the sentence structure. |
        | **Node Color** | Based on the 3rd dimension of the hidden state (h[2]) - a key semantic feature that often correlates with properties like concreteness vs. abstractness, or entity vs. action. Blue/purple nodes typically have different semantic qualities than yellow/green ones. |
        | **Tree Connections** | Show the compositional structure - how meanings combine hierarchically from words to phrases to full sentences. These connections reveal how the Tree-LSTM builds complex meanings from simpler components. |
        | **Span Attribute** | Indicates which tokens (by index) each node covers in the original sentence. For example, a span of [2,5] means this node represents the meaning of words 2 through 4 in the sentence (exclusive of 5). This helps you map tree nodes back to the text. |
        
        ### Color Spectrum Interpretation
        
        - **Purple/Blue** nodes have lower values in the displayed semantic dimension
        - **Green/Yellow** nodes have higher values
        - This dimension may represent features like entity vs. action, concreteness vs. abstractness, or other semantic properties
        
        ### Visual Example
        
        For the sentence "The hungry cat quickly ate the small fish":
        
        - A node at the top represents the entire sentence meaning (S)
        - Nodes below represent "The hungry cat" (NP) and "quickly ate the small fish" (VP)
        - Individual word nodes have different colors based on their semantic properties
        - "Cat" and "fish" (nouns/entities) might have similar colors, while "ate" (action) could have a different color
        - The span [3,4] would indicate the word "cat" (the 4th word, counting from 0)
        - Semantically related words like "cat" and "fish" might appear closer in the semantic space view
        
        ### Reading the Visualizations
        
        1. **Compositional Meaning**
           - Watch how the Tree-LSTM combines smaller meaning units into larger ones
           - Parent nodes integrate and abstract over their children
           - The root node represents the entire sentence meaning
           - Example: See how "hungry" and "cat" combine to form "hungry cat" with properties of both
           
        2. **Semantic Relationships**
           - Nodes that appear close together in the semantic space have similar meanings
           - Distance represents semantic difference or dissimilarity
           - Hovering over nodes reveals the numerical values behind each representation
           - Example: In "river bank", the word "bank" might be positioned closer to "river" than to "bank" in "bank account"
           
        3. **Comparing View Types**
           - Different visualization modes highlight different aspects of meaning
           - The tree view emphasizes grammatical structure (subject-verb-object relationships)
           - The semantic space view reveals meaning relationships that may cross grammatical boundaries
           - Example: In the hybrid view, you can see both how "quickly" modifies "ate" structurally AND their semantic relationship
        """)

    with st.expander("Tree Structure + Semantic Dimensions (Visualization Guide)", expanded=False):
        st.markdown("""
        ## Tree Structure + Semantic Dimensions
        
        **What you're seeing:**
        This visualization shows how meaning is structured within the grammar of a sentence. 
        
        **How it works:**
        - The **X and Y positions** come directly from the first two dimensions of each node's semantic meaning
        - The **Z position** (vertical) shows the grammatical depth in the tree
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
        - Comparisons between phrases at the same grammatical level
        - The relationship between a parent's meaning and its children's meanings
        """)

    with st.expander("Root-Centric View (Visualization Guide)", expanded=False):
        st.markdown("""
        ## Root-Centric View (Top-Down Mode Only)
        
        **What you're seeing:**
        This visualization highlights how information radiates from the root node to all other nodes in the tree during top-down processing.
        
        **How it works:**
        - The **root node** is positioned at the center (0,0,0)
        - Other nodes are arranged in a radial pattern around the root
        - The **distance from center** represents semantic difference from the root
        - The **vertical position** (Z) still represents the grammatical depth
        - **Node sizes** decrease as you move deeper into the tree
        
        **Why it's useful:**
        This approach emphasizes how the top-down process distributes contextual information from the root node
        to all other nodes. It makes it easier to see which parts of the tree receive more influence from the root.
        
        **What to look for:**
        - Nodes clustered near the root are semantically similar to the overall sentence meaning
        - Patterns of how information flows down through the tree levels
        - How different branches of the tree receive different aspects of the root's context
        - The balance between maintaining the root's meaning and developing specialized meaning in branches
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

    # Check if sentence is too long
    if sentence and len(sentence.split()) > MAX_SENTENCE_LENGTH:
        st.warning(f"Sentence is too long ({len(sentence.split())} words). Please use {MAX_SENTENCE_LENGTH} words or fewer to avoid memory issues.")
        sentence = " ".join(sentence.split()[:MAX_SENTENCE_LENGTH])
        st.info(f"Truncated to: '{sentence}...'")

    if sentence:
        try:
            # Create an anchor for the results section
            results_anchor = st.empty()
            
            # Process sentence
            with st.spinner("Building parse tree..."):
                result = encoder.encode(sentence)
                
                # Get POS tags using spaCy
                doc = nlp(sentence)
                pos_tags = {token.i: token.pos_ for token in doc}
            
            # Create tabs for different visualizations - REORDERED to put 3D first
            tab1, tab2, tab3, tab4 = st.tabs([
                "3D Semantic View",
                "Tree Structure", 
                "Similarity Heatmap",
                "Structural Analysis"
            ])
            
            # Display information about the Tree-LSTM mode used
            processing_mode = result.get("mode", "bottom-up")  # Default to bottom-up for backward compatibility
            mode_descriptions = {
                "bottom-up": "Bottom-Up processing: Information flows from leaves to root, capturing compositional meaning",
                "top-down": "Top-Down processing: Information flows from root to leaves, distributing context to parts",
                "both": "Bidirectional processing: Combines both bottom-up and top-down approaches for richer representations"
            }
            
            st.info(mode_descriptions.get(processing_mode, "Standard Tree-LSTM processing"))

            # Add detailed explanation about the different processing modes
            with st.expander("Understanding TreeLSTM Processing Directions", expanded=False):
                st.markdown("""
                ## TreeLSTM Processing Directions Explained

                The direction of information flow in a Tree-LSTM has profound implications for how meaning is constructed and represented. Our `BidirectionalTreeLSTMEncoder` allows you to explore these different approaches. Each direction captures distinct aspects of linguistic structure.

                ### Bottom-Up Processing (Leaves to Root)

                **Our Implementation:** Uses `ChildSumTreeLSTM`.
                
                **How it works:**
                - Information flows from leaf nodes (individual words) upward to the root node (complete sentence).
                - Each parent node's representation is computed by combining information from all its children.
                - The `h` and `c` states stored in each node (and used for visualization) are the direct outputs of this bottom-up pass (`h_bottom_up`, `c_bottom_up`).
                - The root node's final hidden state (`h`) contains a representation of the entire sentence, composed hierarchically.

                **Core Mechanism:**
                - Each node takes inputs from its child nodes (or word embeddings for leaf nodes)
                - The model uses gates (input, output, and forget gates) to control information flow
                - Key innovation: Uses separate forget gates for each child, allowing the node to selectively remember or forget information from each child
                - This selective forgetting is crucial for handling different syntactic and semantic relationships
                - The final representation combines filtered information from all children with the node's own transformation
                
                **Strengths:**
                - Excellent for capturing compositional meaning (how parts combine to create the whole).
                - Naturally reflects syntactic structure and dependencies.
                - Well-suited for tasks like sentiment analysis or text classification where understanding the overall composition is key.

                **Limitations:**
                - Nodes lower in the tree (e.g., individual words) have no direct contextual information from higher-level constituents or the overall sentence meaning during their computation.
                - May struggle with long-distance dependencies if the information is not effectively propagated upwards.

                ### Top-Down Processing (Root to Leaves)

                **Our Implementation:** Uses `TopDownTreeLSTM`.

                **How it works:**
                - Information flows from the root node (representing the whole sentence context) downward to the leaf nodes (individual words).
                - Parent nodes provide contextual information that influences the computation of their children's representations.
                - The `h` and `c` states stored in each node (and used for visualization) are the direct outputs of this top-down pass (`h_top_down`, `c_top_down`).
                - Leaf nodes ultimately receive a representation that incorporates both their local input (e.g., word embedding) and the context passed down from the entire tree.

                **Core Mechanism:**
                - Each node receives context from its parent node (except the root, which starts the process)
                - Similar gate structure (input, output, forget gates) as bottom-up, but used differently
                - Uses a single forget gate to control how much parent information to incorporate
                - The node combines its own input with filtered information from its parent
                - This allows contextual information to flow downward and influence interpretation of parts

                **Strengths:**
                - Excellent for contextual understanding, as it allows the meaning of the whole sentence to influence the interpretation of its parts.
                - Helps in disambiguating word meanings based on the broader sentence context.
                - Well-suited for tasks like conditional text generation or question answering where context is paramount.

                **Limitations:**
                - The initial representation of the root might be less informed before the top-down pass fully propagates context.
                - Can sometimes overemphasize global context at the expense of fine-grained local distinctions if not balanced.

                ### Bidirectional Processing (Combined)

                **Our Implementation:** Leverages both `ChildSumTreeLSTM` and `TopDownTreeLSTM`.

                **How it works:**
                - First, a complete bottom-up pass is performed using `ChildSumTreeLSTM`, yielding `h_bottom_up` and `c_bottom_up` for each node.
                - Then, a complete top-down pass is performed using `TopDownTreeLSTM`, yielding `h_top_down` and `c_top_down` for each node, using the same initial word embeddings.
                - The final hidden state `h` and cell state `c` for each node (used for visualization) are created by **combining** the outputs from both passes. In our current `BidirectionalTreeLSTMEncoder`, this combination is done via element-wise addition:
                    - `node.h = node.h_bottom_up + node.h_top_down`
                    - `node.c = node.c_bottom_up + node.c_top_down`
                - This allows each node's final representation to benefit from both compositional information from its descendants and contextual information from its ancestors and the sentence root.

                **Core Mechanism:**
                - Two separate passes through the tree, each with its own strengths
                - Bottom-up pass captures compositional structure (how parts form wholes)
                - Top-down pass captures contextual influences (how the whole affects parts)
                - Simple addition combines these complementary views for each node
                - Result is a richer representation that benefits from both perspectives

                **Strengths:**
                - Combines the advantages of both bottom-up and top-down approaches, capturing both compositional and contextual semantics.
                - Generally leads to richer and more nuanced representations.
                - Often achieves state-of-the-art performance on a wide range of NLP tasks.

                **Limitations:**
                - Increased computational complexity as it requires two passes over the tree.
                - More parameters in the model if distinct TreeLSTMs are used for each direction.
                - The method of combining the two representations (e.g., addition, concatenation, gating) can influence performance and may need tuning.

                ### Visualization Differences

                The chosen processing direction directly impacts what the `h` vector at each node represents, and thus how the visualizations should be interpreted:

                - **Bottom-Up Mode**: Node vectors `h` are `h_bottom_up`. Visualizations primarily show how meaning is composed from words upwards. Leaf nodes are processed first based on their embeddings, and internal nodes aggregate this information.
                - **Top-Down Mode**: Node vectors `h` are `h_top_down`. Visualizations emphasize how context from the root (entire sentence) is distributed downwards to influence child nodes. The root is initialized, and its state influences its children, and so on.
                - **Bidirectional Mode**: Node vectors `h` are the combination (e.g., sum) of `h_bottom_up` and `h_top_down`. Visualizations reflect this richer representation, where each node's meaning is informed by both its constituents and its context within the larger tree.

                ### Visual Comparison of Information Flow in Our Encoder

                The diagrams below illustrate how the `h` and `c` states are populated in each node based on the selected mode in our `BidirectionalTreeLSTMEncoder`.

                ```
                Bottom-Up (mode='bottom-up')      Top-Down (mode='top-down')        Bidirectional (mode='both')
                (h = h_bottom_up)                 (h = h_top_down)                  (h = h_bottom_up + h_top_down)

                      ↑ h_bu                         ↓ h_td                         ↕ (h_bu, h_td combined)
                     [S]                            [S]                             [S]
                     ↗ ↖                           ↙ ↘                             ↕ ↕
                   [NP] [VP]                      [NP] [VP]                       [NP] [VP]
                   ↗ ↖  ↗ ↖                      ↙ ↘  ↙ ↘                        ↕ ↕  ↕ ↕
                 [D] [N][V] [A]                  [D] [N][V] [A]                  [D] [N][V] [A]
                  ↑   ↑  ↑   ↑                    ↓   ↓  ↓   ↑                    ↕   ↕  ↕   ↕
                 (The)(cat)(is)(fat)            (The)(cat)(is)(fat)              (The)(cat)(is)(fat)

                Processing:                       Processing:                     Processing:
                1. `ChildSumTreeLSTM` pass        1. `TopDownTreeLSTM` pass       1. `ChildSumTreeLSTM` pass (gets h_bu, c_bu)
                2. `h = h_bottom_up`              2. `h = h_top_down`               2. `TopDownTreeLSTM` pass (gets h_td, c_td)
                                                                                  3. `h = h_bu + h_td`
                ```

                ### Analogies to Human Language Processing

                - **Bottom-Up**: Similar to how we might first identify individual words, then combine them into phrases, and finally grasp the overall sentence meaning.
                - **Top-Down**: Akin to how our existing knowledge and the overall context of a conversation or text help us interpret and disambiguate words and phrases as we encounter them.
                - **Bidirectional**: Most closely mirrors sophisticated human language processing, where we simultaneously build meaning from parts and use context to refine our understanding.

                ### Real-World Application Examples

                **Bottom-Up Processing:**
                - **Sentiment Analysis**: Determining whether a review is positive or negative by composing sentiment from individual words and phrases upward.
                - **Text Classification**: Categorizing documents based on the compositional meaning of their contents.
                - **Information Extraction**: Identifying entities and relationships by building up from word-level patterns.

                **Top-Down Processing:**
                - **Word Sense Disambiguation**: Using sentence context to determine correct meaning of ambiguous words (e.g., "bank" as financial institution vs. riverside).
                - **Question Answering**: Contextualizing words and phrases in a passage based on the question being asked.
                - **Text Generation**: Producing coherent text by maintaining global context through the generation process.

                **Bidirectional Processing:**
                - **Machine Translation**: Capturing both local compositional meaning and global context for accurate translations.
                - **Text Summarization**: Understanding both detailed content and overall meaning to create concise summaries.
                - **Language Understanding in Dialogue Systems**: Interpreting user inputs by combining word-level understanding with contextual knowledge.

                Our `BidirectionalTreeLSTMEncoder` provides the flexibility to experiment with these processing strategies. The bidirectional approach, while computationally more intensive, often yields the most comprehensive semantic representations.
                """)

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
                        # Add POS tag for leaf nodes that are individual tokens
                        if span[0] == span[1] - 1 and span[0] < len(doc):
                            pos_tag = pos_tags.get(span[0], "")
                            label = f"{tree['label']}: {token_text} ({pos_tag})"
                        else:
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
                    
                    # Add description based on processing mode
                    if processing_mode == "both":
                        st.markdown("""
                        This 3D visualization shows the tree structure with bidirectional semantic information:
                        * Nodes are color-coded based on their semantic values
                        * The representation combines both bottom-up and top-down processing
                        * This provides a richer view of the sentence structure and meaning
                        """)
                    elif processing_mode == "top-down":
                        st.markdown("""
                        This 3D visualization shows the tree structure with top-down semantic information:
                        * Nodes are color-coded based on their semantic values
                        * Information flows from the root node down to the leaves
                        * This highlights how context is distributed throughout the tree
                        
                        **Try the "Root-Centric View" mode** to see how information radiates from the root to other nodes.
                        In this view, the root node is at the center, and other nodes are arranged by their semantic similarity to the root.
                        """)
                    else:
                        st.markdown("""
                        This 3D visualization shows the tree structure with semantic information:
                        * Nodes are color-coded based on their semantic values
                        * Information flows from the leaves up to the root
                        * This highlights the compositional nature of the sentence
                        """)
                    
                    # Visualization options
                    if processing_mode == "top-down":
                        viz_options = ["Tree Structure + Semantic Dimensions", "Pure Semantic Space (PCA)", "Hybrid View", "Root-Centric View"]
                    else:
                        viz_options = ["Tree Structure + Semantic Dimensions", "Pure Semantic Space (PCA)", "Hybrid View"]
                    
                    viz_mode = st.radio(
                        "Visualization Mode:",
                        viz_options,
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
                            
                        elif viz_mode == "Root-Centric View" and processing_mode == "top-down":
                            # Special visualization for top-down processing
                            # Use the first node (root) as the center
                            if i == 0:  # Root node
                                pos_x = 0  # Center
                                pos_y = 0
                                pos_z = 0
                                size = 20  # Larger size for root
                            else:
                                # Calculate distance from root in vector space 
                                root_vec = all_vectors[0]
                                distance = np.linalg.norm(vec - root_vec)
                                
                                # Use distance for radial positioning
                                angle = i * 2 * np.pi / (len(all_vectors) - 1)  # Distribute around a circle
                                radius = distance * scale * 0.5
                                
                                # Position based on level and angle
                                pos_x = radius * np.cos(angle)
                                pos_y = radius * np.sin(angle)
                                pos_z = -level * 5  # Exaggerate level difference to show hierarchy
                                
                                # Size based on level (smaller as we go deeper)
                                size = 15 - level * 1.5
                                sizes.append(size)
                                
                                # Add edge from parent node
                                if parent_idx >= 0:
                                    x_edges.extend([x[parent_idx], pos_x, None])
                                    y_edges.extend([y[parent_idx], pos_y, None])
                                    z_edges.extend([z[parent_idx], pos_z, None])
                                
                                # Skip the normal edge addition later by continuing
                                x.append(pos_x)
                                y.append(pos_y)
                                z.append(pos_z)
                                labels.append(info['label'])
                                
                                # Determine color based on 3rd semantic dimension
                                color_val = vec[2]
                                colors.append(color_val)
                                
                                continue
                            
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
                        
                        # Use fixed size for all nodes except in Root-Centric view
                        if viz_mode != "Root-Centric View" or i == 0:
                            size = 12  # Fixed size for all nodes
                            sizes.append(size)
                        
                        # Determine color based on 3rd semantic dimension
                        # Use a consistent dimension for coloring
                        color_val = vec[2]
                        colors.append(color_val)
                        
                        # Add edge from parent node if exists and we haven't already done it
                        if parent_idx >= 0 and viz_mode != "Root-Centric View":
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
                        hovertext=[f"{l}<br>Hidden[0:3]=[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]" 
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
                    elif viz_mode == "Root-Centric View":
                        x_title = "Semantic Distance from Root (X)"
                        y_title = "Semantic Distance from Root (Y)"
                        z_title = "Tree Level (Syntax)"
                    else:  # Hybrid view
                        x_title = "Semantic PCA Dimension 1"
                        y_title = "Semantic PCA Dimension 2"
                        z_title = "Tree Level (Syntax)"
                    
                    # Setup camera based on visualization mode
                    if viz_mode == "Root-Centric View":
                        camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0.1, y=0.1, z=2.0)  # Top-down view looking at the root
                        )
                    else:
                        camera = dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=-1.5, z=0.5)
                        )
                    
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(title=x_title, title_font=axis_title_font),
                            yaxis=dict(title=y_title, title_font=axis_title_font),
                            zaxis=dict(title=z_title, title_font=axis_title_font)
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene_camera=camera,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        height=650,  # Add fixed height to ensure proper spacing
                        autosize=True
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
                    
                    # Add description based on processing mode
                    if processing_mode == "both":
                        st.markdown("""
                        This is a traditional top-down tree visualization showing the constituency structure of the sentence.
                        
                        **Bidirectional Processing:**
                        - Bottom-up: Information flows from leaves to root (compositional meaning)
                        - Top-down: Information flows from root to leaves (contextual distribution)
                        - The hidden states combine information from both directions
                        
                        Each node shows the grammatical category (NP, VP, S, etc.) and the span of tokens it covers.
                        """)
                    elif processing_mode == "top-down":
                        st.markdown("""
                        This is a traditional top-down tree visualization showing the constituency structure of the sentence.
                        
                        **Top-Down Processing:**
                        - Information flows from the root down to the leaves
                        - Parent nodes provide context to their children
                        - This method is good for tasks requiring contextual understanding
                        
                        Each node shows the grammatical category (NP, VP, S, etc.) and the span of tokens it covers.
                        """)
                    else:
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
                                # Add POS tag for the token
                                if span[0] < len(doc):
                                    pos_tag = pos_tags.get(span[0], "")
                                    token_text = sentence.split()[span[0]] if span[0] < len(sentence.split()) else ""
                                    label += f"\n→ '{token_text}' ({pos_tag})"
                                else:
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

                    **Part-of-Speech Tags:**
                    - **NOUN**: Nouns (dog, cat, tree)
                    - **VERB**: Verbs (run, eat, sleep)
                    - **ADJ**: Adjectives (big, red, beautiful)
                    - **ADV**: Adverbs (quickly, very, extremely)
                    - **PRON**: Pronouns (I, you, he, she, it)
                    - **DET**: Determiners (the, a, an, this)
                    - **ADP**: Adpositions/Prepositions (in, to, during)
                    - **CCONJ**: Coordinating conjunctions (and, or, but)
                    - **SCONJ**: Subordinating conjunctions (if, while, that)

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
                    if processing_mode == "both":
                        st.markdown("""
                        This visualization shows the hierarchical structure of the parse tree with bidirectional processing.
                        * Bottom-up: Information flows from leaves to root, capturing compositional meaning
                        * Top-down: Information flows from root to leaves, distributing context to parts
                        * Combined: The shown values represent the combined bidirectional representation
                        
                        Colors represent the semantic dimension (3) in the combined representation.
                        """)
                    elif processing_mode == "top-down":
                        st.markdown("""
                        This visualization shows the hierarchical structure of the parse tree with top-down processing.
                        Information flows from the root to the leaves, enabling context distribution from parent nodes to children.
                        
                        Colors represent the semantic dimension (3) in the top-down representation.
                        """)
                    else:
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
                            
                            # Check if this is a leaf node with a single token
                            is_single_token = span[0] == span[1] - 1 and span[0] < len(sentence.split())
                            
                            # Get POS tag for leaf nodes with single tokens
                            pos_tag = ""
                            if is_single_token and span[0] < len(doc):
                                pos_tag = pos_tags.get(span[0], "")
                                if pos_tag:
                                    covered_text = f"{covered_text} ({pos_tag})"
                            
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
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Try a different sentence or check the console for details.") 

# Add vertical space before footer to prevent overlap
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# Add footer with attribution
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; padding: 10px; margin-top: 30px; color: #888;">
    <p>Tree-LSTM Semantic Visualizer | Developed by William Wall</p>
    <p style="font-size: 0.8em;">Built with Streamlit, spaCy, PyTorch, and the Berkeley Neural Parser</p>
    <p style="font-size: 0.8em;">© 2025 | <a href="https://github.com/williamjwall/tree-lstm-generator">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True) 