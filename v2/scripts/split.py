import os
import re
import random

# File paths (Ensure these match where your massive files are saved)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(SCRIPT_DIR, "datasets", "clean")

MASSIVE_TRAIN = os.path.join(CLEAN_DIR, "wiki_clean_train.txt")
MASSIVE_VAL = os.path.join(CLEAN_DIR, "wiki_clean_valid.txt") 

# Final Output paths
SAMPLED_TRAIN = os.path.join(CLEAN_DIR, "wiki_pristine_15k_train.txt")
SAMPLED_VAL = os.path.join(CLEAN_DIR, "wiki_pristine_4k_valid.txt")

# Blocklist for common Hindi words that leak into translated Wikipedia articles
HINDI_BLOCKLIST = {'है', 'और', 'किया', 'में', 'को', 'यह', 'वह', 'गया', 'करते'}

def is_pristine_marathi(sentence):
    # 1. Reject English letters and ANY numbers (English 0-9 or Marathi ०-९)
    if re.search(r'[a-zA-Z0-9०-९]', sentence):
        return False
        
    # 2. Reject Hindi Nuqta (़) - indicates Hindi/Urdu pronunciation
    if re.search(r'\u093C', sentence):
        return False
        
    # 3. Reject known Hindi grammar words
    words = set(sentence.split())
    if words.intersection(HINDI_BLOCKLIST):
        return False
        
    # 4. Strict Unicode Whitelist
    # We ONLY allow: 
    # - Standard Marathi Devanagari block (\u0900-\u093B, \u093D-\u0965, \u0970-\u097F)
    # - Spaces, basic punctuation, quotation marks, and typographic joiners (\u200C, \u200D)
    allowed_pattern = re.compile(r'^[\u0900-\u093B\u093D-\u0965\u0970-\u097F\s\.\,\?\!\|\।\'\"\[\]\(\)\{\}\-\:\;\u200C\u200D\u2018-\u201D]+$')
    
    if not allowed_pattern.match(sentence):
        return False
        
    return True

def purify_and_sample(input_path, output_path, target_count):
    print(f"Purifying {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    pristine_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if is_pristine_marathi(line):
            pristine_lines.append(line + "\n")
            
    print(f"  -> Original sentences: {len(lines)}")
    print(f"  -> Pristine sentences survived: {len(pristine_lines)}")
    
    if len(pristine_lines) < target_count:
        print(f"  ⚠️ Warning: Only {len(pristine_lines)} pristine sentences available.")
        target_count = len(pristine_lines)
        
    # Randomly shuffle and extract the target amount
    sampled_lines = random.sample(pristine_lines, target_count)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)
        
    print(f"✅ Saved {target_count} pristine sentences to {output_path}\n")

# --- EXECUTE ---
purify_and_sample(MASSIVE_TRAIN, SAMPLED_TRAIN, 15000)
purify_and_sample(MASSIVE_VAL, SAMPLED_VAL, 4000)