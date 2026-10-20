import csv
import json
import os
import re
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

LORA_DIR = os.path.join(PROJECT_ROOT, "models", "marathi-gpt-lora")
BASE_MODEL = "l3cube-pune/marathi-gpt"

TEST_FILE = os.path.join(
    PROJECT_ROOT, "datasets", "test", "eval_sentences.jsonl"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DETAILS_FILE = os.path.join(RESULTS_DIR, "evaluation_log.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "evaluation_summary.json")


def normalize_word(word):
    """Normalize a predicted/expected word for comparison."""
    word = word.strip().lower()
    word = re.sub(r"^[\s\"'“”‘’.,!?;:()\[\]{}]+", "", word)
    word = re.sub(r"[\s\"'“”‘’.,!?;:()\[\]{}]+$", "", word)
    return word


def load_test_cases():
    cases = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            required = {"id", "category", "prefix", "target"}
            missing = required - set(item)

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing: {sorted(missing)}"
                )

            cases.append(item)

    if not cases:
        raise ValueError("No evaluation cases found.")

    return cases


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()

    model.to("cpu")
    model.eval()

    print("Model loaded successfully.\n")
    return tokenizer, model


def predict(tokenizer, model, prompt):
    """
    Generate the same style of short continuation used by the
    existing scripts/test_model.py:
      - max_new_tokens=6
      - beam search with 3 beams
      - one result only
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            num_beams=3,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency_ms = (time.perf_counter() - start) * 1000

    input_len = inputs.input_ids.shape[1]

    completion = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    ).strip()

    # Keep the evaluation focused on the actual suggestion.
    for punc in ["|", ".", "?", "!", "\n"]:
        if punc in completion:
            completion = completion.split(punc)[0].strip()

    generated_words = completion.split()
    predicted_first_word = (
        normalize_word(generated_words[0])
        if generated_words
        else ""
    )

    first_two_words = generated_words[:2]

    return {
        "completion": completion,
        "predicted_first_word": predicted_first_word,
        "first_two_words": first_two_words,
        "latency_ms": round(latency_ms, 2),
    }


def evaluate_case(tokenizer, model, case):
    prediction = predict(tokenizer, model, case["prefix"])

    expected = normalize_word(case["target"])
    predicted = prediction["predicted_first_word"]

    top1_correct = predicted == expected

    normalized_two_words = [
        normalize_word(word)
        for word in prediction["first_two_words"]
    ]

    target_in_top2 = expected in normalized_two_words

    return {
        "id": case["id"],
        "category": case["category"],
        "prefix": case["prefix"],
        "expected_next_word": case["target"],
        "predicted_next_word": predicted,
        "generated_suggestion": prediction["completion"],
        "top1_correct": top1_correct,
        "target_in_first_2_words": target_in_top2,
        "latency_ms": prediction["latency_ms"],
    }


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(results)
    top1_correct = sum(r["top1_correct"] for r in results)
    top2_hits = sum(r["target_in_first_2_words"] for r in results)

    average_latency = (
        sum(r["latency_ms"] for r in results) / total
        if total else 0
    )

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "top1_correct": 0,
            "top2_hits": 0,
        }
    )

    for result in results:
        stats = category_stats[result["category"]]
        stats["total"] += 1
        stats["top1_correct"] += int(result["top1_correct"])
        stats["top2_hits"] += int(result["target_in_first_2_words"])

    for stats in category_stats.values():
        stats["top1_accuracy_percent"] = round(
            100 * stats["top1_correct"] / stats["total"], 2
        )
        stats["top2_hit_rate_percent"] = round(
            100 * stats["top2_hits"] / stats["total"], 2
        )

    summary = {
        "task": "Marathi next-word autosuggestion evaluation",
        "total_test_cases": total,
        "top1_correct": top1_correct,
        "top1_accuracy_percent": round(
            100 * top1_correct / total, 2
        ),
        "target_in_first_2_words": top2_hits,
        "target_in_first_2_words_percent": round(
            100 * top2_hits / total, 2
        ),
        "average_generation_latency_ms": round(average_latency, 2),
        "category_results": dict(category_stats),
    }

    with open(DETAILS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "id",
            "category",
            "prefix",
            "expected_next_word",
            "predicted_next_word",
            "generated_suggestion",
            "top1_correct",
            "target_in_first_2_words",
            "latency_ms",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("MARATHI AUTOSUGGESTION MODEL EVALUATION")
    print("=" * 64)
    print(f"Total test cases:                 {total}")
    print(f"Top-1 correct predictions:        {top1_correct}")
    print(f"Top-1 accuracy:                   {summary['top1_accuracy_percent']:.2f}%")
    print(f"Target found in first 2 words:   {top2_hits}")
    print(
        "1-2 word hit rate:                "
        f"{summary['target_in_first_2_words_percent']:.2f}%"
    )
    print(f"Average generation latency:       {average_latency:.2f} ms")
    print("=" * 64)

    print("\nCategory-wise results:")
    for category, stats in category_stats.items():
        print(
            f"- {category}: "
            f"{stats['top1_correct']}/{stats['total']} "
            f"({stats['top1_accuracy_percent']:.2f}%)"
        )

    print("\nFiles created:")
    print(f"- {DETAILS_FILE}")
    print(f"- {SUMMARY_FILE}")


def main():
    print("Starting model evaluation...\n")

    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} held-out test cases.")

    tokenizer, model = load_model()

    results = []

    for number, case in enumerate(test_cases, 1):
        result = evaluate_case(tokenizer, model, case)
        results.append(result)

        status = "PASS" if result["top1_correct"] else "FAIL"

        print(
            f"[{number:02d}/{len(test_cases)}] {status} | "
            f"Expected: {result['expected_next_word']} | "
            f"Predicted: {result['predicted_next_word']} | "
            f"Suggestion: {result['generated_suggestion']!r} | "
            f"{result['latency_ms']:.2f} ms"
        )

    save_results(results)


if __name__ == "__main__":
    main()
