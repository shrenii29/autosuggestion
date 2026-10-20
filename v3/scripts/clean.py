import os
import re
from collections import defaultdict
from indicnlp.tokenize import sentence_tokenize

# 1. Dynamically calculate paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = "C:/Users/Shreni/Desktop/autosuggestion/v3/datasets/txt/MahaSent_All_Test.txt"
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "clean")
LOGS_DIR = os.path.join(PROJECT_ROOT, "datasets", "logs")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(CLEAN_DIR, "MahaSent_All_Test_Clean.txt")
FLAGGED_FILE = os.path.join(LOGS_DIR, "flagged_test.txt")
SIMILAR_FILE = os.path.join(LOGS_DIR, "similar_test.txt")

SIMILARITY_THRESHOLD = 0.75
MAX_WORD_FREQ = 250  # Stops the inverted index from bogging down on words like "आहे"

def split_into_sentences(text):
    """
    Uses Indic NLP to intelligently tokenize Marathi sentences.
    Natively handles initials (एम. एल.), abbreviations, and Dandas.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
        
    sentences = sentence_tokenize.sentence_split(text, lang='mr')
    return [s.strip() for s in sentences if s.strip()]

def clean_formatting(sentence):
    # 1. Strip specified brackets/quotes anywhere in the sentence
    sentence = re.sub(r'["\'\[\]{}()]', '', sentence)
    
    # 2. Collapse multiple spaces into a single space
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # 3. Strip leading punctuation marks from the beginning of the sentence
    # This includes: , . - _ : ; ! ? ( ) [ ] { } । | " ' and standard symbols
    sentence = sentence.lstrip(',.-_勧;:!?()[]{}""\'\'।|•●▪*-+/\\ ')
    
    return sentence.strip()

def is_suspicious(sentence):
    if re.search(r'[a-zA-Z]', sentence):
        return True, "Contains English characters"
        
    if not sentence.endswith(('.', '?', '!', '।', '|')):
        return True, "Missing valid ending punctuation"
        
    words = sentence.split()
    for word in words:
        if len(word) > 16:
            return True, f"Unnaturally long word: {word}"
            
    if re.search(r'\.\.+', sentence) or re.search(r'\?\?+', sentence):
        return True, "Repeating punctuation"
        
    if len(words) < 3:
        return True, "Sentence too short"
        
    return False, ""

def get_word_set(sentence):
    cleaned = re.sub(r'[.?!\।|]', '', sentence)
    return set(cleaned.split())

# --- FAST INVERTED INDEX SIMILARITY CHECK ---
word_index = defaultdict(set)
accepted_sentences = []
accepted_word_sets = []

def is_too_similar_fast(new_word_set):
    if not new_word_set:
        return True, None

    new_len = len(new_word_set)
    candidate_ids = set()
    
    for word in new_word_set:
        if word in word_index:
            if len(word_index[word]) < MAX_WORD_FREQ:
                candidate_ids.update(word_index[word])

    for cand_id in candidate_ids:
        cand_words = accepted_word_sets[cand_id]
        cand_len = len(cand_words)

        if min(new_len, cand_len) / max(new_len, cand_len) < SIMILARITY_THRESHOLD:
            continue

        intersection = len(new_word_set & cand_words)
        union = new_len + cand_len - intersection

        if union > 0 and (intersection / union) >= SIMILARITY_THRESHOLD:
            return True, accepted_sentences[cand_id]

    return False, None



clean_count = 0
flagged_count = 0
similar_count = 0
exact_duplicate_count = 0

seen_exact = set()

print(f"Processing text file: {RAW_DIR}")

# Open output files and input file directly
with open(CLEAN_FILE, "w", encoding="utf-8") as clean_out, \
     open(FLAGGED_FILE, "w", encoding="utf-8") as flagged_out, \
     open(SIMILAR_FILE, "w", encoding="utf-8") as similar_out, \
     open(RAW_DIR, "r", encoding="utf-8", errors="ignore") as infile:
    
    content = infile.read()
    raw_sentences = split_into_sentences(content)
    
    for idx, raw_line in enumerate(raw_sentences, start=1):
        sentence = clean_formatting(raw_line)
        if not sentence:
            continue

        if sentence in seen_exact:
            exact_duplicate_count += 1
            continue
        seen_exact.add(sentence)

        is_bad, reason = is_suspicious(sentence)
        if is_bad:
            flagged_out.write(f"[{reason}] {sentence}\n")
            flagged_count += 1
            continue

        words = get_word_set(sentence)
        too_similar, matched_original = is_too_similar_fast(words)
        
        if too_similar:
            similar_out.write(f"[Similar to: '{matched_original}'] {sentence}\n")
            similar_count += 1
            continue

        curr_id = len(accepted_sentences)
        accepted_sentences.append(sentence)
        accepted_word_sets.append(words)
        
        for w in words:
            if len(word_index[w]) < MAX_WORD_FREQ:
                word_index[w].add(curr_id)

        clean_out.write(sentence + "\n")
        clean_count += 1

        if idx % 5000 == 0:
            print(f"Processed {idx} sentences. ({clean_count} clean sentences so far)")

print("\n--- Summary ---")
print(f"Unique sentences: {clean_count} -> {CLEAN_FILE}")
print(f"Exact duplicates: {exact_duplicate_count}")
print(f"Near-duplicates: {similar_count} -> {SIMILAR_FILE}")
print(f"Flagged: {flagged_count} -> {FLAGGED_FILE}")