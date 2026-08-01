import pandas as pd
import re
from pathlib import Path


# DATA LOADING & PREPROCESSING


print("=== Starting Data Loading & Preprocessing ===\n")

# Set paths
data_dir = Path("Dataset")
output_file = data_dir / "processed_dataset.csv"

all_emails = []   # This holds are emails with their labels

# ---- 1. Load Phishing Emails ----
phishing_folders = ["phishing/phish_01", "phishing/phish_02", "phishing/phish_03", "phishing/kaggle"]

for folder_name in phishing_folders:
    folder = data_dir / folder_name
    if folder.exists():
        print(f"Loading phishing emails from: {folder_name}")
        
        for file in folder.rglob("*.*"):   
            if file.suffix.lower() not in ['.csv', '.txt']:
                continue
                
            try:
                if file.suffix.lower() == '.csv':
                    df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip', low_memory=False)
                    print(f"  → Found CSV: {file.name} | Columns: {list(df.columns)}")
                    
                    # Try different possible columns that contain email body
                    for col in ['message', 'body', 'email', 'payload', 'content', 'text', 'subject']:
                        if col in df.columns:
                            for text in df[col].dropna():
                                if len(str(text).strip()) > 10:   # avoid very short texts
                                    all_emails.append({"text": str(text), "label": 1})
                            print(f"    Added emails from column '{col}'")
                            break
                else:
                    # For .txt files
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if len(text) > 20:
                            all_emails.append({"text": text, "label": 1})
            except Exception as e:
                print(f"    Error reading {file.name}: {e}")

print(f"Total phishing emails loaded: {len([e for e in all_emails if e['label'] == 1])}")

# ---- 2. Load Legitimate Emails ----
legit_file = data_dir / "legitimate" / "emails.csv"

if legit_file.exists():
    print("\nLoading legitimate emails...")
    df_legit = pd.read_csv(legit_file, encoding='latin-1', on_bad_lines='skip')
    
    # Use the correct column (usually 'message' or 'body')
    text_column = 'message' if 'message' in df_legit.columns else 'body'
    legit_texts = df_legit[text_column].dropna().astype(str).tolist()
    
    # Balance: Take roughly 2x number of phishing emails
    num_phishing = len([e for e in all_emails if e['label'] == 1])
    target_legit = min(len(legit_texts), num_phishing * 2)
    
    for text in legit_texts[:target_legit]:
        all_emails.append({"text": text, "label": 0})
    
    print(f"Added {target_legit} legitimate emails for balance")
else:
    print("Warning: Legitimate emails file not found!")

# 3. Create Final DataFrame
df = pd.DataFrame(all_emails)
print(f"\nFinal Dataset Created!")
print(f"Total Emails: {len(df)}")
print(f"Phishing: {df['label'].sum()}")
print(f"Legitimate: {len(df) - df['label'].sum()}")

# Save the raw combined dataset
df.to_csv(output_file, index=False)
print(f"Processed dataset saved to: {output_file}")

print("\n=== Section 2 Completed Successfully! ===")