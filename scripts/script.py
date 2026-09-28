from transformers import AutoTokenizer

# Replace with your exact model path or name
MODEL_NAME = "C:/Users/Shreni/OneDrive/Desktop/AutoSuggestion/models/marathi-gpt-lora"   # or your local fine-tuned path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load your dataset file
FILE_PATH = "C:/Users/Shreni/OneDrive/Desktop/autosuggestion/datasets/clean/clean_raw.txt"

total_tokens = 0
total_sentences = 0

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        tokens = tokenizer.encode(line, add_special_tokens=False)
        total_tokens += len(tokens)
        total_sentences += 1

print("Total sentences:", total_sentences)
print("Total tokens:", total_tokens)
print("Average tokens per sentence:", total_tokens / total_sentences if total_sentences else 0)