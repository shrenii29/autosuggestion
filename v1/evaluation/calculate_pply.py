import os
import csv
import math
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# --- Paths ---
LORA_DIR = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/models/marathi-gpt-lora"
BASE_MODEL = "l3cube-pune/marathi-gpt"
VAL_DATA_PATH = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/datasets/split_data/val.txt"
LOG_FILE = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/evaluation_logs_ppl.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device.upper()}")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.to(device)
model.eval()

# Load validation lines
print(f"Loading validation data from {VAL_DATA_PATH}...")
try:
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit()

if not val_lines:
    print("Validation file is empty.")
    exit()

total_loss = 0.0
total_tokens = 0

print(f"\nEvaluating Perplexity across {len(val_lines)} validation sentences...")

with torch.no_grad():
    for line in tqdm(val_lines):
        encodings = tokenizer(line, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        
        # Causal LMs require at least 2 tokens to compute a valid next-token shift
        if input_ids.shape[1] < 2:
            continue
            
        target_ids = input_ids.clone()
        
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        
        # In causal LM, loss is averaged over (seq_len - 1) tokens due to shifting
        num_valid_tokens = input_ids.shape[1] - 1
        
        total_loss += neg_log_likelihood.item() * num_valid_tokens
        total_tokens += num_valid_tokens

if total_tokens == 0:
    print("No valid sequences found for evaluation.")
    exit()

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print("\n" + "=" * 40)
print("📊 VALIDATION PERPLEXITY RESULTS")
print("=" * 40)
print(f"Total Evaluated Tokens: {total_tokens}")
print(f"Mean Cross-Entropy Loss: {avg_loss:.4f}")
print(f"Perplexity (PPL):        {perplexity:.2f}")
print("=" * 40)

# --- CSV Logging ---
file_exists = os.path.isfile(LOG_FILE)
with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Timestamp", "Model_Dir", "Total_Tokens", "Val_Loss", "Perplexity"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = os.path.basename(LORA_DIR)
    writer.writerow([timestamp, model_name, total_tokens, f"{avg_loss:.4f}", f"{perplexity:.2f}"])

print(f"📝 PPL logged to {LOG_FILE}")