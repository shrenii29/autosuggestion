import os
import time
import math
import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths to models and test data
BASE_MODEL = "l3cube-pune/marathi-gpt"
LORA_DIR = os.path.join(PROJECT_ROOT, "proxy","models", "marathi-gpt-lora") # Update if your LoRA path is different
TEST_FILE = os.path.join(PROJECT_ROOT, "proxy","datasets", "clean", "wiki_pristine_2k_test.txt")

# Set to "cpu" for strict production latency profiling, or "cuda" for fast accuracy testing
DEVICE = "cpu"  
TOP_K = 3

# ==========================================
# 2. STATS & METRICS HELPERS
# ==========================================
def calc_95_ci(successes, n):
    """Calculates the 95% Confidence Interval for a binomial proportion."""
    if n == 0: return 0.0, 0.0
    p = successes / n
    ci = 1.96 * math.sqrt((p * (1 - p)) / n)
    return p * 100, ci * 100

def clean_word(word):
    """Strips punctuation for strict exact-match comparison."""
    return re.sub(r'[.?!\।|\[\]\(\)\,\'\"\-]', '', word).strip()

# ==========================================
# 3. CORE DECODING ENGINE
# ==========================================
def evaluate_position(model, tokenizer, context, prefix, target_word, device):
    """
    Evaluates a single keystroke state using a single forward pass.
    Handles Marathi subwords perfectly by stitching the token before evaluating.
    """
    # 1. Setup exact user input state
    input_text = context + (" " + prefix if prefix else "")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # 2. Single Forward Pass (Latency Profiled)
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids)
    forward_latency = (time.perf_counter() - start_time) * 1000

    # 3. Get Distribution for the NEXT token
    next_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)
    
    # Grab top 50 candidates to filter through
    _, top_indices = torch.topk(probs, 50)
    
    suggestions = []
    target_clean = clean_word(target_word)
    
    for idx in top_indices:
        # 4. Word-Boundary Resolution
        # We append the token ID to the context and decode the whole thing.
        # This prevents GPT-2 BPE artifact bugs (where "माहि" + "ती" gets mangled).
        candidate_ids = torch.cat([input_ids[0], torch.tensor([idx.item()], device=device)])
        candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)
        
        # Extract the last word generated
        words = candidate_text.split()
        if not words: continue
        predicted_word = clean_word(words[-1])
        
        # 5. Prefix Enforcement (If mid-word, it MUST start with what the user typed)
        if prefix and not predicted_word.startswith(prefix):
            continue
            
        if predicted_word and predicted_word not in suggestions:
            suggestions.append(predicted_word)
            
        if len(suggestions) >= TOP_K:
            break

    # 6. Scoring
    is_top1 = 1 if len(suggestions) > 0 and suggestions[0] == target_clean else 0
    is_top3 = 1 if target_clean in suggestions else 0

    return is_top1, is_top3, forward_latency

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def run_evaluation():
    print(f"Loading Base Model ({BASE_MODEL}) and LoRA Adapters...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load and merge weights for 0-latency architecture
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
    model.to(DEVICE)
    model.eval()
    
    if DEVICE == "cpu":
        torch.set_num_threads(4) # Pin threads to avoid context switching
        
    print(f"Loading Test Set: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    # Limit to 500 sentences for this run to keep it manageable, adjust as needed
    test_sentences = sentences[:500] 
    
    print(f"\nEvaluating {len(test_sentences)} sentences on {DEVICE.upper()}...")
    
    # Trackers
    next_word_stats = {"n": 0, "top1": 0, "top3": 0, "latencies": []}
    mid_word_stats = {"n": 0, "top1": 0, "top3": 0, "latencies": []}

    for i, sentence in enumerate(test_sentences):
        words = sentence.split()
        
        # Evaluate every word position in the sentence (skipping the first word)
        for j in range(1, len(words)):
            context = " ".join(words[:j])
            target = words[j]
            target_clean = clean_word(target)
            
            if not target_clean or len(target_clean) < 2:
                continue # Skip single letters or pure punctuation
                
            # SCENARIO A: Next-Word Prediction (User just hit space)
            t1, t3, lat = evaluate_position(model, tokenizer, context, prefix="", target_word=target, device=DEVICE)
            next_word_stats["n"] += 1
            next_word_stats["top1"] += t1
            next_word_stats["top3"] += t3
            next_word_stats["latencies"].append(lat)
            
            # SCENARIO B: Mid-Word Completion (User typed half the word)
            # We simulate typing the first half of the word
            prefix_len = max(1, len(target_clean) // 2)
            prefix = target_clean[:prefix_len]
            
            t1_m, t3_m, lat_m = evaluate_position(model, tokenizer, context, prefix, target_word=target, device=DEVICE)
            mid_word_stats["n"] += 1
            mid_word_stats["top1"] += t1_m
            mid_word_stats["top3"] += t3_m
            mid_word_stats["latencies"].append(lat_m)

        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i + 1}/{len(test_sentences)} sentences...")

    # ==========================================
    # 5. GENERATE FINAL REPORT
    # ==========================================
    print("\n" + "="*50)
    print("📊 MARATHI AUTOCOMPLETE EVALUATION REPORT")
    print("="*50)
    
    for label, stats in [("NEXT-WORD PREDICTION (Word Bank)", next_word_stats), 
                         ("MID-WORD COMPLETION (Ghost Text)", mid_word_stats)]:
                         
        top1_pct, top1_ci = calc_95_ci(stats["top1"], stats["n"])
        top3_pct, top3_ci = calc_95_ci(stats["top3"], stats["n"])
        
        lats = stats["latencies"]
        mean_lat = np.mean(lats)
        p50_lat = np.percentile(lats, 50)
        p95_lat = np.percentile(lats, 95)
        
        print(f"\n{label}")
        print("-" * 40)
        print(f"Sample Size (N)   : {stats['n']} prediction points")
        print(f"Top-1 Accuracy    : {top1_pct:.2f}% (±{top1_ci:.2f}%)")
        print(f"Top-3 Accuracy    : {top3_pct:.2f}% (±{top3_ci:.2f}%)")
        print(f"Forward Latency   : {mean_lat:.2f} ms (Mean) | {p50_lat:.2f} ms (P50) | {p95_lat:.2f} ms (P95)")

if __name__ == "__main__":
    run_evaluation()