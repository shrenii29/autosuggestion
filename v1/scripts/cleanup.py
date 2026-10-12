import os
import re

# 1. Dynamically calculate paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Steps up one level to the main folder

# Define the folder paths
RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "clean")
LOGS_DIR = os.path.join(PROJECT_ROOT, "datasets", "logs")

# 2. Automatically create the clean and logs folders if they don't exist
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the exact file paths
INPUT_FILE = os.path.join(RAW_DIR, "raw_gemini-3.1-flash-lite-preview.txt")
CLEAN_FILE = os.path.join(CLEAN_DIR, "marathi_clean_3.1flp0.2.txt")
FLAGGED_FILE = os.path.join(LOGS_DIR, "marathi_flagged2.txt")
SIMILAR_FILE = os.path.join(LOGS_DIR, "marathi_similar2.txt")


SIMILARITY_THRESHOLD = 0.60  # Flag if 60% or more words overlap

def clean_formatting(sentence):
    sentence = re.sub(r'["\'\[\]{}()]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()

def is_suspicious(sentence):
    # Contains Latin letters or numbers (0-9)
    if re.search(r'[a-zA-Z0-9]', sentence):
        return True, "Contains English characters or numbers"
        
    # Valid ending punctuation
    if not sentence.endswith(('.', '?', '!')):
        return True, "Missing valid ending punctuation"
        
    # Unnaturally long compound word
    words = sentence.split()
    for word in words:
        if len(word) > 16:
            return True, f"Unnaturally long word detected: {word}"
            
    # Repeating punctuation
    if re.search(r'\.\.+', sentence) or re.search(r'\?\?+', sentence):
        return True, "Repeating punctuation"
        
    return False, ""

def get_word_set(sentence):
    # Strip trailing punctuation for clean token comparison
    cleaned = re.sub(r'[.?!\।]', '', sentence)
    return set(cleaned.split())

def is_too_similar(new_word_set, accepted_word_sets):
    if not new_word_set:
        return True, None
        
    for accepted_sentence, accepted_words in accepted_word_sets:
        intersection = len(new_word_set & accepted_words)
        union = len(new_word_set | accepted_words)
        
        if union == 0:
            continue
            
        similarity = intersection / union
        if similarity >= SIMILARITY_THRESHOLD:
            return True, accepted_sentence
            
    return False, None

print(f"Reading raw data from: {INPUT_FILE}")

clean_count = 0
flagged_count = 0
similar_count = 0
exact_duplicate_count = 0

seen_exact = set()
accepted_word_sets = []

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(CLEAN_FILE, "w", encoding="utf-8") as clean_out, \
     open(FLAGGED_FILE, "w", encoding="utf-8") as flagged_out, \
     open(SIMILAR_FILE, "w", encoding="utf-8") as similar_out:
    
    for raw_line in infile:
        sentence = clean_formatting(raw_line)
        if not sentence:
            continue

        # 1. Exact Duplicate Check (after cleaning)
        if sentence in seen_exact:
            exact_duplicate_count += 1
            continue
        seen_exact.add(sentence)

        # 2. Rule-based Quality Checks
        is_bad, reason = is_suspicious(sentence)
        if is_bad:
            flagged_out.write(f"[{reason}] {sentence}\n")
            flagged_count += 1
            continue

        # 3. Near-Duplicate / Similarity Check
        words = get_word_set(sentence)
        too_similar, matched_original = is_too_similar(words, accepted_word_sets)
        
        if too_similar:
            similar_out.write(f"[Similar to: '{matched_original}'] {sentence}\n")
            similar_count += 1
            continue

        # Validated unique sentence
        accepted_word_sets.append((sentence, words))
        clean_out.write(sentence + "\n")
        clean_count += 1

print("\nCleanup Complete!")
print(f"✅ Pristine, unique sentences saved: {clean_count} -> {CLEAN_FILE}")
print(f"🗑️ Exact duplicates removed: {exact_duplicate_count}")
print(f"🔄 Near-duplicates flagged: {similar_count} -> {SIMILAR_FILE}")
print(f"⚠️ Rule violations quarantined: {flagged_count} -> {FLAGGED_FILE}")