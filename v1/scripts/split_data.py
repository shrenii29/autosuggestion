import os
import random

# Dynamically route to the folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "clean")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "datasets", "split_data")

# Create the split_data folder if it doesn't exist
os.makedirs(SPLIT_DIR, exist_ok=True)

# Define exact file paths
INPUT_FILE = os.path.join(CLEAN_DIR, "clean_raw.txt")
TRAIN_FILE = os.path.join(SPLIT_DIR, "train.txt")
VAL_FILE = os.path.join(SPLIT_DIR, "val.txt")

# 90% Training, 10% Validation
SPLIT_RATIO = 0.80 

print(f"Reading cleaned data from: {INPUT_FILE}")

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Read all non-empty lines
        lines = [line.strip() for line in f if line.strip()]
        
    # CRITICAL: Shuffle the dataset to mix the 5 categories evenly
    random.seed(42) # Fixed seed ensures the exact same split if you run it twice
    random.shuffle(lines)
    
    # Calculate the split point
    split_index = int(len(lines) * SPLIT_RATIO)
    
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]
    
    # Write to the new files inside the split_data folder
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
        
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
        
    print("\nDataset Split Complete!")
    print(f"Total Sentences: {len(lines)}")
    print(f"Training Set:   {len(train_lines)} sentences -> datasets/split_data/train.txt")
    print(f"Validation Set: {len(val_lines)} sentences -> datasets/split_data/val.txt")

except FileNotFoundError:
    print(f"\nError: Could not find {INPUT_FILE}")
    print("Ensure the cleanup script finished successfully before running this.")