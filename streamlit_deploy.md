# Streamlit Deployment Notes

## Tree-LSTM Semantic Visualizer

This app visualizes constituency parse trees of sentences using Tree-LSTM networks and interactive 3D visualizations.

### Deployment Notes

When deploying this app on Streamlit Cloud:

1. **First run will take extra time** - the app needs to download several models:
   - spaCy English language model
   - Berkeley Neural Parser model
   - BERT model and tokenizer

2. **Memory usage** - This app uses Tree-LSTM networks and 3D visualizations that require significant memory. For best results:
   - Keep sentences under 60 words
   - Avoid using multiple complex visualizations simultaneously
   - Refresh the page if performance degrades

3. **Troubleshooting**:
   - If models fail to download, try refreshing the page
   - If visualizations appear blank, try using a different browser
   - For persistent issues, check the logs or contact the developer

### Model Usage

The app uses:
- spaCy with the Berkeley Neural Parser for constituency parsing
- BERT for token embeddings
- A Child-Sum Tree-LSTM for semantic composition

### Visualization Guide

The app provides multiple views of the same sentence:
1. 3D Tree Visualization - Shows semantic dimensions and tree structure
2. 2D Tree Visualization - Traditional constituency parse tree view  
3. Semantic Similarity Heatmap - Shows relationships between constituents
4. Phrasal Structure Diagram - Compact hierarchical visualization 