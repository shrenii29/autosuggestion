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

OUTPUT_FILE = "raw_gemini-3.1-flash-lite-preview.txt"
BATCH_SIZE = 200 

categories = [
    {"name": "Workplace & Email", "count": 2000},
    {"name": "Casual Chat & Social", "count": 2000},
    {"name": "Inquiries & General Questions", "count": 2000},
    {"name": "Opinions, Feedback & Reviews", "count": 1000},
    {"name": "Starters & Connectors", "count": 1000}
]

# Random seeds to prevent the LLM from repeating the same outputs across batches
scenarios = [
    # General Everyday
    "discussing travel plans", "asking about health", "scheduling a meeting", 
    "complaining about traffic", "sharing food preferences", "tech support",
    "making excuses", "wishing someone well", "following up on a task",
    "asking for directions", "talking about weather", "discussing family",
    "giving a quick update", "asking for a favor", "expressing urgency",
    
    # Tech, Studies & Professional
    "debugging a coding error", "requesting a deadline extension", "discussing a research paper",
    "planning a tech presentation", "asking for project feedback", "troubleshooting software bugs",
    "coordinating an internship interview", "discussing hardware or IoT circuits",
    "sharing an update on a college assignment", "asking for help with data analysis",

    # Hobbies, Interests & Leisure
    "discussing a badminton match", "talking about a favorite rock band", 
    "sharing thoughts on a sci-fi book or space exploration", "planning a trip to a heritage site", 
    "discussing home renovation plans", "preparing for a public speaking event",
    "coordinating a weekend meetup with a friend", "discussing an upcoming concert",

    # Logistics & Minor Emergencies
    "apologizing for being late", "reminding someone of an appointment", 
    "asking to borrow a laptop charger", "reporting a power cut or internet outage", 
    "explaining a missed call", "checking if a store is open", "finding a parking spot",

    # Opinions, Shopping & Inquiries
    "reviewing a new software tool", "complaining about a faulty product", 
    "asking for movie or series recommendations", "negotiating a price at a shop", 
    "discussing local street food", "inquiring about a gym or club membership",
    "comparing two different mobile phones", "asking for a recipe"
]

def generate_batch(category_name, size):
    # Pick a random scenario to force the LLM down a new creative path
    current_scenario = random.choice(scenarios)
    
    prompt = f"""
    You are creating training data for a mobile keyboard autocomplete model in Marathi.
    Generate a JSON list containing exactly {size} unique, gramatically correct sentences for the category: "{category_name}".
    Do not create similar or almost similar sentences at all.

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
            model='gemini-3.1-flash-lite-preview',
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