import os
import json
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the GEMINI_API_KEY from your .env file
load_dotenv()

client = genai.Client()

OUTPUT_FILE = "raw_sentences_gemini-flash-lite-latest.txt"
BATCH_SIZE = 200 

categories = [
    {"name": "Workplace & Email", "count": 1800},
    {"name": "Casual Chat & Social", "count": 1800},
    {"name": "Inquiries & General Questions", "count": 1200},
    {"name": "Opinions, Feedback & Reviews", "count": 600},
    {"name": "Starters & Connectors", "count": 600}
]

# Random seeds to prevent the LLM from repeating the same outputs across batches
scenarios = [
    "discussing travel plans", "asking about health", "scheduling a meeting", 
    "complaining about traffic", "sharing food preferences", "tech support",
    "making excuses", "wishing someone well", "following up on a task",
    "asking for directions", "talking about weather", "discussing family",
    "giving a quick update", "asking for a favor", "expressing urgency"
]

def generate_batch(category_name, size):
    # Pick a random scenario to force the LLM down a new creative path
    current_scenario = random.choice(scenarios)
    
    prompt = f"""
    You are creating training data for a mobile keyboard autocomplete model in Marathi.
    Generate a JSON list containing exactly {size} unique, gramatically correct sentences for the category: "{category_name}".
    
    Current Scenario Focus: {current_scenario}
    
    CRITICAL RULES:
    1. Length: Every sentence MUST be between 3 and 15 words. Short, natural mobile typing.
    2. Perspective: Prioritize First and Second Person ("मी", "आम्ही", "तू", "तुम्ही").
    3. Content: Completely avoid third-person historical facts, news items, or encyclopedic definitions.
    4. Diversity: Use broad vocabulary. Sound like a real person typing.
    5. Script: Write exclusively in clean, grammatically correct Devanagari script.
    
    Return ONLY a valid JSON array of strings.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1.0 
            )
        )
        sentences = json.loads(response.text)
        if isinstance(sentences, list):
            return [s.strip() for s in sentences if 3 <= len(s.split()) <= 15]
    except Exception as e:
        print(f"\nAPI Error: {e}")  
    return []

print("Starting dynamic dataset generation...")
with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for cat in categories:
        collected = 0
        target = cat["count"]
        print(f"\nGenerating {target} sentences for: {cat['name']}")
        
        while collected < target:
            rem = target - collected
            current_batch_size = min(BATCH_SIZE, rem)
            
            batch = generate_batch(cat["name"], current_batch_size)
            
            # --- SAFETY NET ---
            if not batch:
                print("\nAPI overloaded or failed. Waiting 10 seconds before retrying...")
                time.sleep(10)
                continue
            
            for sentence in batch:
                f.write(sentence + "\n")
                collected += 1
                
            print(f"Progress: {collected}/{target} sentences saved.", end="\r")
            time.sleep(5) # Normal 1-second pause between successful batches

print(f"\nDataset successfully generated and saved to {OUTPUT_FILE}!")