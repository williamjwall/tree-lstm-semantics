#!/usr/bin/env python3
"""
Helper script to switch between model implementations.
This lets the user choose between:
- model.py (default): Uses BertTokenizerFast with word_ids() method
- model_alt.py (alternative): Works with any tokenizer by processing tokens individually
"""

import os
import sys
import shutil

def backup_file(filename):
    """Create a backup of the file if it exists."""
    if os.path.exists(filename):
        backup = f"{filename}.bak"
        print(f"Backing up {filename} to {backup}")
        shutil.copy(filename, backup)

def enable_alternative():
    """Switch to use the alternative implementation."""
    src_dir = os.path.join("src", "tree_lstm_viz")
    model_file = os.path.join(src_dir, "model.py")
    alt_file = os.path.join(src_dir, "model_alt.py")
    
    if not os.path.exists(alt_file):
        print(f"Error: Alternative model file {alt_file} not found.")
        return

    # Backup the current model file
    backup_file(model_file)
    
    # Copy the alternative to the main file
    print("Enabling alternative implementation...")
    shutil.copy(alt_file, model_file)
    print("Done! The application will now use the alternative implementation.")

def restore_original():
    """Restore the original model implementation."""
    src_dir = os.path.join("src", "tree_lstm_viz")
    model_file = os.path.join(src_dir, "model.py")
    backup = f"{model_file}.bak"
    
    if not os.path.exists(backup):
        print(f"Error: Backup file {backup} not found. Cannot restore.")
        return
    
    # Restore from backup
    print("Restoring original implementation...")
    shutil.copy(backup, model_file)
    print("Done! The application will now use the original implementation.")
    
if __name__ == "__main__":
    print("Tree-LSTM Model Switcher")
    print("-----------------------")
    print("1. Use alternative implementation (no fast tokenizer required)")
    print("2. Restore original implementation (requires fast tokenizer)")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        enable_alternative()
    elif choice == "2":
        restore_original()
    else:
        print("Exiting without changes.")
        
    print("\nRemember to restart your application for changes to take effect.") 