# Tree-LSTM Semantic Visualizer

An interactive tool for visualizing compositional semantics through constituency parse trees and Tree-LSTM networks. This project includes visualization capabilities for bottom-up, top-down, and bidirectional Tree-LSTM models.

You can also view it on this link: https://lstm-tree-generator.streamlit.app/

(note I am not paying for premium lol, so might take a few minutes for the instance to 'wake up')

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
  - Pure Semantic Space (PCA)
  - Tree Structure + PCA
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

- **Pure Semantic Space (PCA)**: Visualizes semantic relationships in pure 3D space using PCA
- **Tree Structure + PCA**: Balances grammatical structure with semantic relationships using PCA-reduced dimensions
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

## Streamlit Cloud (hosted deploy)

This app is meant to run on [Streamlit Community Cloud](https://lstm-tree-generator.streamlit.app/) (or your own Cloud app). You do **not** need to run it locally.

**What Cloud does automatically**

- `pip install -r requirements.txt` (including `transformers<5`, which Benepar needs)
- `packages.txt` system packages (graphviz, build tools)
- Python version from `runtime.txt` (`python-3.10`)

**What Cloud does *not* run**

- `setup.sh` — models are downloaded when the app starts (parser + BERT on the loading screen)

**After you fix the repo**

1. Push changes to the GitHub repo connected to your Cloud app.
2. Wait for the deploy to finish (Manage app → logs).
3. Open the app; the **first load after deploy or wake-up** can take several minutes while models download.
4. Free tier: keep sentences short; bidirectional mode uses more RAM. If the app reboots, it may need to download again.

**If you still see “No constituency parse”**

- Confirm `requirements.txt` contains `transformers>=4.30.0,<5.0.0` and redeploy.
- Reboot the app from the Cloud dashboard (⋮ → Reboot app).
- Check deploy logs for `transformers` 5.x or Benepar download errors.

## Troubleshooting

- **Memory Issues**: For systems with limited RAM, try shorter sentences
- **Parsing Errors**: Complex sentences may cause parser issues; try simpler alternatives
- **Missing Libraries**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **"No constituency parse" / Benepar init failed**: Requires `transformers<5` in `requirements.txt` on Streamlit Cloud; parser model downloads on first app start
- **`ModuleNotFoundError: No module named 'torchvision'`**: Add `torchvision>=0.15.0` to `requirements.txt` and redeploy (Streamlit's watcher imports optional transformers vision modules)
- **`Cannot copy out of meta tensor; no data!` (Benepar)**: Use CPU PyTorch wheels and `transformers<4.46` in `requirements.txt` (see current file); redeploy and reboot the app

## License

MIT License - See LICENSE file for details.

## Author

[William Wall](https://github.com/williamjwall)

## Project Structure

```
tree-lstm-generator/
├── streamlit_app.py          # Main application
├── src/
│   ├── tree_lstm_viz/        # Tree-LSTM implementations
│   │   ├── model.py          # Bottom-up TreeLSTM (original)
│   │   ├── top_down_model.py # Top-down TreeLSTM
│   │   ├── model_bidirectional.py # Bidirectional TreeLSTM
│   │   ├── benepar_utils.py  # Berkeley Parser utilities
│   │   └── logger.py         # Structured logging
│   ├── visualization/        # Visualization utilities
│   └── utils/                # Helper utilities
├── setup.sh                  # Installation script
└── requirements.txt          # Dependencies
``` 

## References and Further Reading

- Tai, K. S., Socher, R., & Manning, C. D. (2015). [Improved semantic representations from tree-structured long short-term memory networks](https://aclanthology.org/P15-1150.pdf). ACL.
- Zhang, X., Lu, L., & Lapata, M. (2016). [Top-down tree long short-term memory networks](https://aclanthology.org/N16-1035.pdf). NAACL. 
