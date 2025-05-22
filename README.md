# Tree-LSTM Semantic Visualizer

## Overview

This project provides an intuitive interface for exploring how distributed semantic representations are composed in Tree-structured LSTMs, combining:

- Constituency parsing via spaCy/Benepar
- Contextualized embeddings via BERT
- Compositional semantics via multiple TreeLSTM architectures:
  - Bottom-up (Child-Sum Tree-LSTM)
  - Top-down Tree-LSTM
  - Bidirectional Tree-LSTM (combined approach)
- Interactive 2D and 3D visualizations that adapt to each processing direction

## Key Features

- **Multiple Processing Directions**:
  - Bottom-up: Information flows from leaves to root (composition of parts)
  - Top-down: Information flows from root to leaves (contextual distribution)
  - Bidirectional: Combines both approaches for richer representations
- **Interactive 3D Visualization** with multiple views:
  - Tree Structure + Semantic Dimensions
  - Pure Semantic Space (PCA)
  - Hybrid View
  - Root-Centric View (for top-down processing)
- **Semantic Insights**: Node position, size, and color convey semantic information
- **Comparative Analysis**: Explore how different processing directions affect semantic representations
- **CPU Compatible**: No GPU required

## Quick Start

```bash
# Clone repository
git clone https://github.com/williamjwall/tree-lstm-generator.git
cd tree-lstm-generator

# Setup (creates virtual environment and installs dependencies)
./setup.sh

# Run the application
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```

## Usage

1. Enter a sentence in the text input field
2. Select the processing direction (bottom-up, top-down, or bidirectional)
3. Explore the visualizations using the tabs:
   - **3D Semantic View**: Interactive 3D representation of semantic space
   - **Tree Structure**: Traditional constituency parse tree
   - **Similarity Heatmap**: Shows semantic relationships between nodes
   - **Structural Analysis**: Detailed tree structure information

## Processing Direction Selection

The application lets you choose from three processing modes:
- **Bottom-Up**: Traditional approach where leaf nodes (words) are processed first and information flows upward to the root (complete sentence)
- **Top-Down**: Innovative approach where the root node is processed first and contextual information flows downward to the leaves
- **Bidirectional**: Combined approach that performs both passes and merges the information, creating richer representations

## Example Sentences

- **Simple**: "The cat sat on the mat."
- **Complex**: "Although he was very tired, the old man walked five miles to reach the nearest town."
- **Question**: "How many students are taking the advanced linguistics course this semester?"
- **Ambiguous**: "The man who hunts ducks out on weekends."
- **Garden Path**: "The cotton clothing is made of grows in Mississippi."

## Visualization Modes

- **Tree Structure + Semantic Dimensions**: Shows grammatical structure with semantic dimensions for X and Y positions
- **Pure Semantic Space (PCA)**: Visualizes semantic relationships in pure 3D space using PCA
- **Hybrid View**: Balances grammatical structure with semantic relationships
- **Root-Centric View** (for top-down mode): Shows how information radiates from the root node

## Technical Details

The processing pipeline:

1. **Parsing**: Sentence → constituency tree (spaCy/Benepar)
2. **Token Embedding**: Contextual representations (BERT)
3. **Processing Direction**:
   - Bottom-up: Leaf-to-root traversal using Child-Sum Tree-LSTM
   - Top-down: Root-to-leaf traversal using Top-Down Tree-LSTM
   - Bidirectional: Both passes with representation combining

## Implementation

- **ChildSumTreeLSTM**: Implements the bottom-up approach with separate forget gates for each child
- **TopDownTreeLSTM**: Implements the top-down approach with a single forget gate for parent influence
- **BidirectionalTreeLSTMEncoder**: Combines both approaches through element-wise addition

## Troubleshooting

- **Memory Issues**: For systems with limited RAM, try shorter sentences
- **Parsing Errors**: Complex sentences may cause parser issues; try simpler alternatives
- **Missing Libraries**: Ensure all dependencies are installed (`pip install -r requirements.txt`)

## License

MIT License - See LICENSE file for details.

## Author

[William Wall](https://github.com/williamjwall)
