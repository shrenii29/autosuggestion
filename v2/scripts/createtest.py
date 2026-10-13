import os
import random
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(SCRIPT_DIR, "datasets", "clean")

# The data we already sampled (The Lockout Bank)
SAMPLED_TRAIN = os.path.join(CLEAN_DIR, "wiki_pristine_15k_train.txt")
SAMPLED_VAL = os.path.join(CLEAN_DIR, "wiki_pristine_4k_valid.txt")

# The massive file we will draw the test set from
MASSIVE_VAL = os.path.join(CLEAN_DIR, "wiki_clean_valid.txt") 

# The Final Output
TEST_OUT = os.path.join(CLEAN_DIR, "wiki_pristine_2k_test.txt")

HINDI_BLOCKLIST = {'है', 'और', 'किया', 'में', 'को', 'यह', 'वह', 'गया', 'करते'}
TEST_SET_SIZE = 2000

def is_pristine_marathi(sentence):
    if re.search(r'[a-zA-Z0-9०-९]', sentence):
        return False
    if re.search(r'\u093C', sentence):
        return False
    words = set(sentence.split())
    if words.intersection(HINDI_BLOCKLIST):
        return False
    allowed_pattern = re.compile(r'^[\u0900-\u093B\u093D-\u0965\u0970-\u097F\s\.\,\?\!\|\।\'\"\[\]\(\)\{\}\-\:\;\u200C\u200D\u2018-\u201D]+$')
    if not allowed_pattern.match(sentence):
        return False
    return True

def build_lockout_bank():
    lockout_set = set()
    for filepath in [SAMPLED_TRAIN, SAMPLED_VAL]:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    lockout_set.add(line.strip())
    print(f"🔒 Lockout Bank initialized with {len(lockout_set)} active train/val sentences.")
    return lockout_set

def generate_blind_test_set():
    lockout_bank = build_lockout_bank()
    print(f"Scanning {MASSIVE_VAL} for unseen test candidates...")
    
    with open(MASSIVE_VAL, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    valid_test_candidates = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 1. Must be pristine
        if not is_pristine_marathi(line):
            continue
            
        # 2. MUST NOT exist in Train or Val sets
        if line in lockout_bank:
            continue
            
        valid_test_candidates.append(line + "\n")
        
    print(f"Found {len(valid_test_candidates)} pristine, completely unseen sentences.")
    
    if len(valid_test_candidates) < TEST_SET_SIZE:
        raise ValueError(f"Not enough data! Only found {len(valid_test_candidates)} valid sentences for testing.")
        
    sampled_test = random.sample(valid_test_candidates, TEST_SET_SIZE)
    
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        f.writelines(sampled_test)
        
    print(f"✅ Blind Test Set created: {TEST_SET_SIZE} sentences saved to {TEST_OUT}")

generate_blind_test_set()