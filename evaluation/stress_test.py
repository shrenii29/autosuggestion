import time
import statistics
import torch
import torch.ao.quantization
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

torch.set_num_threads(4)

LORA_DIR = r"C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/models/marathi-gpt-lora"
BASE_MODEL = "l3cube-pune/marathi-gpt"

print("Loading model for CPU Stress Test...")
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.to("cpu")
model.eval()

print("Compressing model weights to INT8...")
model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

test_phrases = [
    "मला", "मी आज", "तुझे नाव", "तो काय", "आम्ही उद्या",
    "हे काम", "पाऊस पडत", "तू कुठे", "माझे घर", "काल रात्री"
] * 5 

print("\nRunning CPU Warm-up...")
warmup_inputs = tokenizer("मला", return_tensors="pt").to("cpu")
with torch.no_grad():
    model.generate(**warmup_inputs, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)

latencies = []

print("\n🚀 Commencing Rapid-Fire Stress Test (50 consecutive requests)...")
print("-" * 50)

for i, phrase in enumerate(test_phrases):
    inputs = tokenizer(phrase, return_tensors="pt").to("cpu")
    start_time = time.perf_counter() 
    
    with torch.no_grad():
        # --- ALGORITHM SHIFT: Greedy Search ---
        # Removing num_beams forces the model to take the single fastest, most confident path
        model.generate(
            **inputs,
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id
        )
        
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)
    
    if (i + 1) % 10 == 0:
        print(f"Request {i+1}/50 processed... (Latest: {latency_ms:.2f} ms)")

avg_latency = statistics.mean(latencies)
median_latency = statistics.median(latencies)
p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
min_latency = min(latencies)
max_latency = max(latencies)

print("\n" + "=" * 45)
print("⏱️  GREEDY ALGORITHM LATENCY STRESS TEST")
print("=" * 45)
print(f"Total Requests: {len(test_phrases)}")
print(f"Target Budget:  50.00 - 80.00 ms")
print("-" * 45)
print(f"Average:      {avg_latency:.2f} ms")
print(f"Median (P50): {median_latency:.2f} ms")
print(f"P95 Latency:  {p95_latency:.2f} ms")
print(f"Fastest:      {min_latency:.2f} ms")
print(f"Slowest:      {max_latency:.2f} ms")
print("=" * 45)

if p95_latency <= 80:
    print("\n✅ SUCCESS: The engine comfortably meets the strict latency budget under load!")
elif avg_latency <= 80:
    print("\n⚠️ BORDERLINE: Average is good, but occasional lag spikes exceed 80ms.")
else:
    print("\n❌ FAILED: We need a smaller base model.")