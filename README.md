# Tree-LSTM Semantic Visualizer

**NOTE** takes a few minutes for the app to launch, but try it out here: lstm-tree-generator.streamlit.app/


An interactive tool for visualizing compositional semantics through constituency parse trees and Tree-LSTM networks. This project includes capabilities for both visualization and training custom Tree-LSTM models.

## Overview

This project provides an intuitive interface for exploring how distributed semantic representations are composed in Tree-structured LSTMs, combining:

- Constituency parsing via spaCy/Benepar
- Contextualized embeddings via BERT
- Compositional semantics via Child-Sum Tree-LSTM
- Interactive 2D and 3D visualizations

## Key Features

- **Interactive 3D Visualization** with multiple views:
  - Tree Structure + Semantic Dimensions
  - Pure Semantic Space (PCA)
  - Hybrid View
- **Semantic Insights**: Node position, size, and color convey semantic information
- **Model Training**: Train custom Tree-LSTM models on your own datasets
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
2. Explore the visualizations using the tabs:
   - **3D Semantic View**: Interactive 3D representation of semantic space
   - **Tree Structure**: Traditional constituency parse tree
   - **Similarity Heatmap**: Shows semantic relationships between nodes
   - **Structural Analysis**: Detailed tree structure information

## Example Sentences

- **Simple**: "The cat sat on the mat."
- **Complex**: "Although he was very tired, the old man walked five miles to reach the nearest town."
- **Question**: "How many students are taking the advanced linguistics course this semester?"

## Technical Details

The processing pipeline:

1. **Parsing**: Sentence → constituency tree (spaCy/Benepar)
2. **Token Embedding**: Contextual representations (BERT)
3. **Composition**: Bottom-up tree traversal (Tree-LSTM)

## Troubleshooting

- **Memory Issues**: For systems with limited RAM, try shorter sentences
- **Parsing Errors**: Complex sentences may cause parser issues; try simpler alternatives
- **Missing Libraries**: Ensure all dependencies are installed (`pip install -r requirements.txt`)

## License

MIT License - See LICENSE file for details.

## Author

[William Wall](https://github.com/williamjwall)
``` 
