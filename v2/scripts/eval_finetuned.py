import os
import time
import math
import re
import csv
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Adjust if needed

BASE_MODEL = "l3cube-pune/marathi-gpt"
# Pointing to the V2 folder where the training script just saved your weights
LORA_DIR = r"C:\Users\Shreni\OneDrive\Desktop\proxy\models\marathi-gpt-lora-v4"

TEST_FILE = r"C:\Users\Shreni\OneDrive\Desktop\proxy\datasets\clean\val.txt"
CSV_LOG_FILE = r"C:\Users\Shreni\OneDrive\Desktop\proxy\finetuned_benchmark_results.csv"

DEVICE = "cpu"  # Keep CPU for accurate apples-to-apples latency comparison
TOP_K = 3
MAX_TEST_SENTENCES = 500  # Must match what you used for the baseline

# ==========================================
# 2. STATS & LOGGING HELPERS
# ==========================================
def calc_95_ci(successes, n):
    if n == 0: return 0.0, 0.0
    p = successes / n
    ci = 1.96 * math.sqrt((p * (1 - p)) / n)
    return p * 100, ci * 100

def clean_word(word):
    return re.sub(r'[.?!\।|\[\]\(\)\,\'\"\-]', '', word).strip()

def log_to_csv(run_name, scenario, stats):
    file_exists = os.path.isfile(CSV_LOG_FILE)
    
    top1_pct, top1_ci = calc_95_ci(stats["top1"], stats["n"])
    top3_pct, top3_ci = calc_95_ci(stats["top3"], stats["n"])
    
    lats = stats["latencies"]
    mean_lat = np.mean(lats) if lats else 0
    p50_lat = np.percentile(lats, 50) if lats else 0
    p95_lat = np.percentile(lats, 95) if lats else 0
    max_lat = np.max(lats) if lats else 0

    with open(CSV_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Run_Name", "Model", "Device", "Scenario", 
                             "Sample_N", "Top1_Acc", "Top1_CI", "Top3_Acc", "Top3_CI", 
                             "Lat_Mean_ms", "Lat_P50_ms", "Lat_P95_ms", "Lat_Max_ms"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name, BASE_MODEL, DEVICE.upper(), scenario,
            stats["n"], 
            round(top1_pct, 2), round(top1_ci, 2), 
            round(top3_pct, 2), round(top3_ci, 2),
            round(mean_lat, 2), round(p50_lat, 2), round(p95_lat, 2), round(max_lat, 2)
        ])

# ==========================================
# 3. CORE DECODING ENGINE
# ==========================================
def evaluate_position(model, tokenizer, context, prefix, target_word, device):
    input_text = context + (" " + prefix if prefix else "")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids)
    forward_latency = (time.perf_counter() - start_time) * 1000

    next_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)
    
    _, top_indices = torch.topk(probs, 50)
    
    suggestions = []
    target_clean = clean_word(target_word)
    
    for idx in top_indices:
        candidate_ids = torch.cat([input_ids[0], torch.tensor([idx.item()], device=device)])
        candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)
        
        words = candidate_text.split()
        if not words: continue
        predicted_word = clean_word(words[-1])
        
        if prefix and not predicted_word.startswith(prefix):
            continue
            
        if predicted_word and predicted_word not in suggestions:
            suggestions.append(predicted_word)
            
        if len(suggestions) >= TOP_K:
            break

    is_top1 = 1 if len(suggestions) > 0 and suggestions[0] == target_clean else 0
    is_top3 = 1 if target_clean in suggestions else 0

    return is_top1, is_top3, forward_latency

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def run_finetuned_evaluation():
    print(f"Loading Tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # THE BIG DIFFERENCE: We load the base model, then fuse your trained weights into it.
    print(f"Loading Base Model and merging LoRA weights from {LORA_DIR}...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
    model.to(DEVICE)
    model.eval()
    
    if DEVICE == "cpu":
        torch.set_num_threads(4) 
        
    print(f"Loading Blind Test Set: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    if MAX_TEST_SENTENCES:
        sentences = sentences[:MAX_TEST_SENTENCES]
    
    print(f"\nEvaluating {len(sentences)} sentences on {DEVICE.upper()} (WITH FINE-TUNED WEIGHTS)...")
    
    next_word_stats = {"n": 0, "top1": 0, "top3": 0, "latencies": []}
    mid_word_stats = {"n": 0, "top1": 0, "top3": 0, "latencies": []}

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        
        for j in range(1, len(words)):
            context = " ".join(words[:j])
            target = words[j]
            target_clean = clean_word(target)
            
            if not target_clean or len(target_clean) < 2:
                continue 
                
            # SCENARIO A: Next-Word Prediction
            t1, t3, lat = evaluate_position(model, tokenizer, context, "", target, DEVICE)
            next_word_stats["n"] += 1
            next_word_stats["top1"] += t1
            next_word_stats["top3"] += t3
            next_word_stats["latencies"].append(lat)
            
            # SCENARIO B: Mid-Word Completion
            prefix_len = max(1, len(target_clean) // 2)
            prefix = target_clean[:prefix_len]
            
            t1_m, t3_m, lat_m = evaluate_position(model, tokenizer, context, prefix, target, DEVICE)
            mid_word_stats["n"] += 1
            mid_word_stats["top1"] += t1_m
            mid_word_stats["top3"] += t3_m
            mid_word_stats["latencies"].append(lat_m)

        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i + 1}/{len(sentences)} sentences...")

    # Automatically grab the folder name (e.g., "marathi-gpt-lora-v3") 
    model_version = os.path.basename(LORA_DIR)
    
    # Log to CSV with the specific version name
    log_to_csv(f"{model_version}_gemini_test", "Next-Word", next_word_stats)
    log_to_csv(f"{model_version}_gemini_test", "Mid-Word", mid_word_stats)

    print("\n" + "="*50)
    print("✅ FINE-TUNED EVALUATION COMPLETE")
    print(f"Results appended to: {CSV_LOG_FILE}")
    print("="*50)
    
    t1_pct, _ = calc_95_ci(next_word_stats["top1"], next_word_stats["n"])
    print(f"Fine-Tuned Next-Word Top-1 Accuracy: {t1_pct:.2f}%")

if __name__ == "__main__":
    run_finetuned_evaluation()