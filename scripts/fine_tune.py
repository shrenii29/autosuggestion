import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TRAIN_FILE = os.path.join(PROJECT_ROOT, "datasets", "split_data", "train.txt")
VAL_FILE = os.path.join(PROJECT_ROOT, "datasets", "split_data", "val.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "marathi-gpt-lora")

MODEL_ID = "l3cube-pune/marathi-gpt"

print(f"Loading Base Model: {MODEL_ID}")

# 2. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Explicitly add a new padding token since the base model lacks one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Resize the model's vocabulary to accept the new [PAD] token
model.resize_token_embeddings(len(tokenizer))

# 3. Configure LoRA
# GPT-2 architecture uses 'c_attn' for its attention mechanism
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,               # Rank: Keeps the adapter tiny and fast
    lora_alpha=16,     # Scaling factor
    lora_dropout=0.1, 
    target_modules=["c_attn"] 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load & Tokenize Dataset
print("\nPreparing Datasets...")
dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

def tokenize_function(examples):
    # Truncate to maximum of 32 tokens (perfect for short mobile sentences)
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=32, 
        padding="max_length"
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=3e-4,           # Slightly higher LR works well for LoRA
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,           # 3 epochs is usually the sweet spot to prevent overfitting
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"              # Disables Weights & Biases logging
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # The collator automatically shifts labels by 1 for next-word prediction
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 7. Start Training
print("\nStarting Training...")
trainer.train()

# 8. Save the final adapter weights
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete! LoRA weights saved to: {OUTPUT_DIR}")