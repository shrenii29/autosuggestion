import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LORA_DIR = r"C:/Users/Shreni/OneDrive/Desktop/autosuggestion/models/marathi-gpt-lora"
BASE_MODEL = "l3cube-pune/marathi-gpt"
FILE = "workspace.txt"

open(FILE, 'a', encoding="utf-8").close()

print("Booting up Gmail-style phrase completion...")
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.to("cpu")
model.eval()

print("\n✅ Engine Online. Type in workspace.txt and hit save.")

last_mtime = 0
last_processed_text = ""

while True:
    try:
        current_mtime = os.path.getmtime(FILE)
        if current_mtime != last_mtime:
            last_mtime = current_mtime
            
            with open(FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            user_input = content.split("\n---")[0].strip()
            
            if not user_input or user_input == last_processed_text:
                continue
                
            start_time = time.time()
            inputs = tokenizer(user_input, return_tensors="pt").to("cpu")
            
            with torch.no_grad():
                # Gmail style: Generate a short continuous phrase
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=6,       # Allow enough tokens for 2-3 words
                    num_beams=3,            # High mathematical accuracy
                    num_return_sequences=1, # Only return the single most logical continuation
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            input_len = inputs.input_ids.shape[1]
            
            # Extract the generated sequence
            completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            # Stop the completion immediately if it hits sentence-ending punctuation
            for punc in ["|", ".", "?", "!", "\n"]:
                if punc in completion:
                    completion = completion.split(punc)[0]
            
            latency = (time.time() - start_time) * 1000
            
            # Rewrite the file to show the inline autocomplete
            with open(FILE, "w", encoding="utf-8") as f:
                f.write(f"{user_input}\n")
                f.write("-" * 30 + "\n")
                f.write("Smart Compose:\n")
                f.write(f"{user_input} {completion.strip()}\n")
                f.write(f"(Latency: {latency:.2f} ms)\n")
                
            last_processed_text = user_input
            last_mtime = os.path.getmtime(FILE) # Ignore the script's own save event
            
    except FileNotFoundError:
        pass
        
    time.sleep(0.3)