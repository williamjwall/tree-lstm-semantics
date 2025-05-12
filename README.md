# Tree-LSTM Semantic Visualizer

An interactive visualization tool that demonstrates compositional semantics through constituency parse trees and Tree-LSTM networks. This project now includes capabilities for training custom Tree-LSTM models on your own datasets.

> **Note:** After running the application, consider taking a screenshot of your visualization to replace this note with an actual image of the tool in action!

## Overview

This project provides an intuitive interface for exploring how distributed semantic representations are composed in Tree-structured LSTMs. It combines:

- Constituency parsing (via spaCy/Benepar)
- Contextualized token embeddings (via BERT)
- Compositional semantics (via Child-Sum Tree-LSTM)
- Interactive 2D and 3D visualizations
- Model training capabilities for custom datasets

## Features

- **Constituency Parse Trees**: Visualize the syntactic structure of sentences
- **Semantic Composition**: See how meaning is built bottom-up through the tree
- **Interactive 3D Visualization**: Multiple views showing semantic relationships:
  - Tree Structure + Semantic Dimensions: See the syntax and semantics together
  - Pure Semantic Space (PCA): Experience semantic similarity in 3D space
  - Hybrid View: Balance between structural and semantic representation
- **Semantic Insights**: Node size, color, and position all convey semantic information
- **Model Training**: Train custom Tree-LSTM models on your own datasets
- **CPU Compatible**: No GPU required, runs on standard hardware

## Installation

### Prerequisites

- Python 3.8+
- pip
- virtualenv (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/williamjwall/tree-lstm-generator.git
   cd tree-lstm-generator
   ```

2. Run the setup script which creates a virtual environment and installs dependencies:
   ```bash
   ./setup.sh
   ```
   
   The setup script will:
   - Create a Python virtual environment
   - Install required Python packages
   - Download necessary language models
   - Set up the application structure

### Manual Setup

If the setup script doesn't work for you, follow these steps:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required models:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import benepar; benepar.download('benepar_en3')"
   ```

## Usage

### Visualization Mode

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser at the URL shown in the terminal (typically http://localhost:8501)

4. Enter a sentence in the text input field or use the default example

5. Explore both the 2D and 3D visualizations using the tabs

### Training Mode

1. Prepare your dataset in the required format (see `data/README.md` for details)

2. Configure your training parameters in `config/training_config.yaml`

3. Run the training script:
   ```bash
   python train.py
   ```

4. Monitor training progress in the logs and TensorBoard

5. Use the trained model for visualization:
   ```bash
   streamlit run streamlit_app.py --model_path path/to/your/model
   ```

## Example Sentences

Try these example sentences to explore different semantic and syntactic phenomena:

### Simple Sentences
- "The cat sat on the mat."
- "John gave Mary a book yesterday."

### Complex Sentences
- "The old man who lived by the sea told fascinating stories about his adventures."
- "Although it was raining heavily, the children decided to go outside and play."

### Ambiguous Sentences
- "I saw the man with the telescope." (Who has the telescope?)
- "The cotton clothing is made of grows in Mississippi." (Garden path sentence)

### Semantic Contrasts
- "The bank by the river was flooded." vs. "The bank on Main Street was robbed."
- "Time flies like an arrow; fruit flies like a banana."

### Interesting Structure
- "Julia kindly gave milk to a very friendly new neighbor after going to the river bank."
- "Colorless green ideas sleep furiously." (Syntactically correct but semantically unusual)

## Technical Details

### Model Architecture

This project implements a Child-Sum Tree-LSTM as described in the paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075) by Tai et al. (2015).

The processing pipeline works as follows:

1. **Parsing**: The sentence is parsed into a constituency tree using spaCy with the Benepar parser
2. **Token Embedding**: Each token is embedded using BERT to obtain contextual representations
3. **Composition**: The Tree-LSTM composes these embeddings bottom-up through the tree:
   - Leaf nodes receive token embeddings
   - Internal nodes combine child representations using the Tree-LSTM cell
   - The root node contains the representation of the entire sentence

### Training Process

The model can be trained on custom datasets with the following features:

- **Data Loading**: Supports various data formats through the `DataLoader` class
- **Training Loop**: Implements a flexible training loop with:
  - Custom loss functions
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
- **Evaluation**: Includes comprehensive evaluation metrics:
  - Semantic similarity scores
  - Tree structure preservation metrics
  - Compositional accuracy

### Tree-LSTM Cell

The Child-Sum Tree-LSTM variant adapts the standard LSTM cell to tree structures:

- Each node has input, output, and update gates (i, o, u)
- Each node has a separate forget gate (f) for each child
- The hidden state for a node is computed by combining the hidden states of its children

The key equations implemented are:

```
h̃ = sum(h_k) for all children k
i = σ(W_i * x_j + U_i * h̃ + b_i)
o = σ(W_o * x_j + U_o * h̃ + b_o)
u = tanh(W_u * x_j + U_u * h̃ + b_u)
f_k = σ(W_f * x_j + U_f * h_k + b_f) for each child k
c_j = i ⊙ u + sum(f_k ⊙ c_k) for all children k
h_j = o ⊙ tanh(c_j)
```

## Visualization Details

### 2D Tree Visualization

The traditional tree view uses Graphviz to render the constituency parse tree with:
- Internal nodes showing phrase labels and spans
- Leaf nodes showing tokens and their positions
- Top-down hierarchical layout

### 3D Semantic Visualization

The 3D view has three modes, each providing a different perspective on the data:

1. **Tree Structure + Semantic Dimensions**:
   - X, Y: First two dimensions of the hidden state vectors
   - Z: Tree depth (syntax structure)
   - Shows how semantic values change within syntactic structure

2. **Pure Semantic Space (PCA)**:
   - Uses Principal Component Analysis (PCA) to reduce 768D vectors to 3D
   - Shows semantic relationships without constraints from tree structure
   - Similar meaning nodes cluster together in space

3. **Hybrid View**:
   - X, Y: Semantic similarity from PCA
   - Z: Tree level information
   - Balances semantic and syntactic information

4. **Training Progress View**:
   - Visualizes model training progress
   - Shows loss curves and metrics
   - Displays semantic space evolution during training

Each node's:
- **Position**: Reflects semantic content and/or tree structure
- **Size**: Based on the magnitude of its semantic vector
- **Color**: Based on its semantic value along a key dimension
- **Hover text**: Shows exact values of the first few hidden state dimensions

## Troubleshooting

### Common Issues

#### Tokenizer Error
If you encounter errors related to the tokenizer:
```
Error: Error processing tree: word_ids() is not available when using non-fast tokenizers
```

**Solution**: Use the alternative model implementation that doesn't rely on fast tokenizers:
```bash
python use_alternative_model.py
```

#### CUDA/GPU Issues
The application is designed to run on CPU mode to avoid compatibility issues. If you're facing GPU-related errors:

**Solution**: The code already forces CPU mode, but you can double-check with:
```python
import torch
print(torch.cuda.is_available())  # Should print False
```

#### Parsing Errors
For very long or complex sentences, the parser might fail:

**Solution**: Try simpler sentences or break down complex ones into shorter units.

#### Memory Issues
For systems with limited RAM:

**Solution**: Close other applications and try sentences of moderate length.

#### Missing Libraries
If you encounter import errors:

**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Alternative Implementations

If you're experiencing issues with the tokenizer, you can switch to an alternative implementation that doesn't require fast tokenizers:

```bash
python use_alternative_model.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Tree-LSTM implementation is based on the paper by Tai et al. (2015)
- Visualization techniques inspired by various semantic space visualization tools
- Built using Streamlit, spaCy, PyTorch, and Plotly
- Training infrastructure inspired by modern deep learning frameworks

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{tree-lstm-generator,
  author = {William Wall},
  title = {Tree-LSTM Semantic Visualizer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/williamjwall/tree-lstm-generator}
}
```

## Author

[William Wall](https://github.com/williamjwall)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change. 

tree-lstm-generator/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── setup.sh
├── src/
│   ├── __init__.py
│   ├── tree_lstm/
│   │   └── __init__.py
│   ├── visualization/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── assets/
├── models/
└── venv/ 