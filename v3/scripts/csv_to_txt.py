import csv

# Define your paths
INPUT_CSV = "C:/Users/Shreni/Desktop/autosuggestion/v3/datasets/MahaSent_All_Val.csv"
OUTPUT_TXT = "C:/Users/Shreni/Desktop/autosuggestion/v3/datasets/txt/MahaSent_All_Val.txt"

TEXT_COLUMN = "text"  # Replace with your actual text column header name
LABEL_COLUMN = "label"    # Replace with your actual label column header name

print(f"Filtering rows where {LABEL_COLUMN} is 0 or 1...")

count_0 = 0
count_1 = 0

with open(INPUT_CSV, mode="r", encoding="utf-8", errors="ignore") as csv_in, \
     open(OUTPUT_TXT, mode="w", encoding="utf-8") as txt_out:
    
    reader = csv.DictReader(csv_in)
    
    for row in reader:
        # Get the label value as a stripped string
        current_label = row.get(LABEL_COLUMN, "").strip()
        
        # Accept if the label is either '0' or '1'
        if current_label in ("0", "1"):
            text_data = row.get(TEXT_COLUMN, "").strip()
            
            if text_data:
                txt_out.write(text_data + "\n")
                
                # Keep track of counts for the summary
                if current_label == "0":
                    count_0 += 1
                else:
                    count_1 += 1

print("\n--- Extraction Summary ---")
print(f"Rows with label '0': {count_0}")
print(f"Rows with label '1': {count_1}")
print(f"Total sentences extracted: {count_0 + count_1} -> {OUTPUT_TXT}")
