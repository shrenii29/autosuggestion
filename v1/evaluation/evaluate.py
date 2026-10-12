import os
import csv
import string
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# --- Paths ---
LORA_DIR = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/models/marathi-gpt-lora"
BASE_MODEL = "l3cube-pune/marathi-gpt"
VAL_DATA_PATH = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/datasets/split_data/val.txt"
LOG_FILE = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/evaluation_logs.csv"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.to("cpu")
model.eval()

print(f"Loading validation data from {VAL_DATA_PATH}...")
try:
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        # Let's test 100 sentences first so you don't have to wait 5 minutes on CPU
        test_samples = [line.strip() for line in f if line.strip()][:100]
except Exception as e:
    print(f"\n❌ ERROR READING FILE: {e}")
    exit()

top_1_hits = 0
top_3_hits = 0
total_valid_tests = 0

# Helper function to remove punctuation from a word
def clean_word(word):
    return word.translate(str.maketrans('', '', string.punctuation)).strip()

print("\nRunning True Top-K Autocomplete Evaluation...")
for text in tqdm(test_samples):
    words = text.strip().split()
    if len(words) < 3:
        continue 
        
    input_text = " ".join(words[:-1])
    target_word = clean_word(words[-1])
    
    if not target_word:
        continue
        
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        # Let it finish the word just like the real app
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    suggestions = []
    input_len = inputs.input_ids.shape[1]
    
    for seq in outputs:
        new_text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        word = clean_word(new_text.split(" ")[0]) if new_text else ""
        
        if word and word not in suggestions:
            suggestions.append(word)
            
    if len(suggestions) > 0:
        total_valid_tests += 1
        
        # Check Top 1
        if target_word == suggestions[0]:
            top_1_hits += 1
            
        # Check Top 3
        if target_word in suggestions:
            top_3_hits += 1

top_1_accuracy = (top_1_hits / total_valid_tests) * 100
top_3_accuracy = (top_3_hits / total_valid_tests) * 100

print("\n" + "="*40)
print("🎯 TRUE EVALUATION RESULTS")
print("="*40)
print(f"Total Sentences Tested: {total_valid_tests}")
print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
print(f"Top-3 Accuracy: {top_3_accuracy:.2f}%")
print("="*40)

# --- CSV Logging ---
file_exists = os.path.isfile(LOG_FILE)
with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Timestamp", "Model_Dir", "Total_Tests", "Top_1_Acc_Percent", "Top_3_Acc_Percent"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = os.path.basename(LORA_DIR)
    writer.writerow([timestamp, model_name, total_valid_tests, f"{top_1_accuracy:.2f}", f"{top_3_accuracy:.2f}"])