# tree_lstm_viz package
try:
    # Try to import from model (uses fast tokenizer)
    from .model import TreeLSTMEncoder, Node, build_tree, ChildSumTreeLSTM
except ImportError as e:
    # Fallback to the alternative implementation
    from .model_alt import TreeLSTMEncoder, Node, build_tree, ChildSumTreeLSTM
    print("Using alternative model implementation that doesn't require fast tokenizers") 