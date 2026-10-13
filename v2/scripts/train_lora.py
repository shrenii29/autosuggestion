import os
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "clean")

TRAIN_FILE = r"C:\Users\Shreni\OneDrive\Desktop\proxy\datasets\clean\wiki_pristine_15k_train.txt"
VAL_FILE = r"C:\Users\Shreni\OneDrive\Desktop\proxy\datasets\clean\wiki_pristine_4k_valid.txt"
OUTPUT_DIR = r"C:\Users\Shreni\OneDrive\Desktop\proxy\models\marathi-gpt-lora-v4"

BASE_MODEL = "l3cube-pune/marathi-gpt"


# ==========================================
# 2. CUSTOM LOGGING CALLBACK
# ==========================================
class LossLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val_loss = metrics.get("eval_loss", 0)
        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
        print(f"\n📊 Epoch {state.epoch:.2f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

# ==========================================
# 3. TOKENIZER & BASE MODEL INITIALIZATION
# ==========================================
print(f"Loading tokenizer and base model ({BASE_MODEL})...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# THE V2 FIX (That we know works): Forcefully add brand new tokens.
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# ==========================================
# 4. DATASET PROCESSING
# ==========================================
print("Loading and tokenizing datasets...")
dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

def tokenize_function(examples):
    # Append the newly created [EOS] token to teach boundaries
    texts_with_eos = [text.strip() + f" {tokenizer.eos_token}" for text in examples["text"]]
    
    outputs = tokenizer(
        texts_with_eos, 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    )
    
    labels = []
    for input_ids in outputs["input_ids"]:
        # THE CRITICAL FIX: Mask the padding tokens with -100
        # It dynamically looks up whatever ID [PAD] was assigned.
        masked_label = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in input_ids]
        labels.append(masked_label)
        
    outputs["labels"] = labels
    return outputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ==========================================
# 5. HIGH-IMPACT LORA SETUP
# ==========================================
lora_config = LoraConfig(
    r=32,                     
    lora_alpha=64,            
    target_modules=["c_attn", "c_proj", "c_fc"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out=True
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 6. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",          
    save_strategy="epoch",          
    logging_strategy="steps",
    logging_steps=50,
    
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  
    fp16=True,                      
    
    # High-impact learning rate to shift the 124M model's weights
    learning_rate=2e-3,             
    weight_decay=0.01,
    warmup_ratio=0.1,               
    lr_scheduler_type="cosine",     
    num_train_epochs=5,             
    
    load_best_model_at_end=True,    
    metric_for_best_model="eval_loss",
    report_to="none"                
)

# ==========================================
# 7. TRAINER EXECUTION
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[LossLoggingCallback()]
)

print("\n🚀 Starting LoRA V5 Fine-Tuning...")
trainer.train()

print(f"\n✅ Training Complete. Best model saved to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)