import os
import re
from collections import defaultdict
from indicnlp.tokenize import sentence_tokenize

# 1. Dynamically calculate paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = r"C:\Users\Shreni\OneDrive\Desktop\proxy\marathi-wikipedia-articles\versions\1\valid\valid"
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "clean")
LOGS_DIR = os.path.join(PROJECT_ROOT, "datasets", "logs")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(CLEAN_DIR, "wiki_clean_valid.txt")
FLAGGED_FILE = os.path.join(LOGS_DIR, "wiki_flagged_valid.txt")
SIMILAR_FILE = os.path.join(LOGS_DIR, "wiki_similar_valid.txt")

SIMILARITY_THRESHOLD = 0.60
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
    sentence = re.sub(r'["\'\[\]{}()]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()

def is_suspicious(sentence):
    if re.search(r'[a-zA-Z]', sentence):
        return True, "Contains English characters"
        
    if not sentence.endswith(('.', '?', '!', '।', '|')):
        return True, "Missing valid ending punctuation"
        
    words = sentence.split()
    for word in words:
        if len(word) > 16:
            return True, f"Unnaturally long word detected: {word}"
            
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


# --- MAIN EXECUTION ---
print(f"Scanning directory: {RAW_DIR}")

clean_count = 0
flagged_count = 0
similar_count = 0
exact_duplicate_count = 0

seen_exact = set()

all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
total_files = len(all_files)
print(f"Found {total_files} text files to process.")

with open(CLEAN_FILE, "w", encoding="utf-8") as clean_out, \
     open(FLAGGED_FILE, "w", encoding="utf-8") as flagged_out, \
     open(SIMILAR_FILE, "w", encoding="utf-8") as similar_out:
    
    for file_idx, filename in enumerate(all_files, start=1):
        file_path = os.path.join(RAW_DIR, filename)
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
            content = infile.read()
            raw_sentences = split_into_sentences(content)
            
            for raw_line in raw_sentences:
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

        if file_idx % 2000 == 0 or file_idx == total_files:
            print(f"Processed {file_idx}/{total_files} files... ({clean_count} clean sentences so far)")

print("\nCleanup Complete!")
print(f"✅ Pristine, unique sentences saved: {clean_count} -> {CLEAN_FILE}")
print(f"🗑️️ Exact duplicates removed: {exact_duplicate_count}")
print(f"🔄 Near-duplicates flagged: {similar_count} -> {SIMILAR_FILE}")
print(f"⚠️ Rule violations quarantined: {flagged_count} -> {FLAGGED_FILE}")