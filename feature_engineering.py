import pandas as pd
import re
from pathlib import Path

print("=== Feature Engineering with Regex ===\n")

# Load the processed dataset from Section 2
data_dir = Path("Dataset")
input_file = data_dir / "processed_dataset.csv"
output_file = data_dir / "final_features_dataset.csv"

if not input_file.exists():
    print("Error: processed_dataset.csv not found! Please run build_dataset.py first.")
    exit()

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} emails for feature engineering.")

# ===================== FEATURE EXTRACTION FUNCTION =====================
def extract_features(text):
    if not isinstance(text, str):
        text = ""
    
    features = {}
    
    # 1. Basic Text Statistics
    features['length'] = len(text)
    features['num_exclamation'] = text.count('!')
    features['num_question'] = text.count('?')
    features['num_caps'] = sum(1 for c in text if c.isupper())
    features['num_dollar'] = text.count('$')
    features['num_at'] = text.count('@')
    
    # 2. URL Related Features
    urls = re.findall(r'http[s]?://[^\s<>"{}|\\^`\[\]]+', text)
    features['num_urls'] = len(urls)
    features['has_ip_url'] = 1 if any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', u) for u in urls) else 0
    
    # 3. Urgent / Suspicious Keywords
    urgent_words = ['urgent', 'immediately', 'verify', 'suspend', 'login', 'update', 'expire', 
                    'now', 'final notice', 'action required', 'click here', 'reset password']
    features['urgent_count'] = sum(text.lower().count(word) for word in urgent_words)
    
    features['has_free'] = 1 if 'free' in text.lower() else 0
    features['has_winner'] = 1 if any(w in text.lower() for w in ['winner', 'prize', 'won']) else 0
    
    return pd.Series(features)

# Apply feature extraction
print("Extracting features from emails... (this may take 1-3 minutes)")
features_df = df['text'].apply(extract_features)

# Combine with original label
final_df = pd.concat([df[['text', 'label']], features_df], axis=1)

print(f"Feature extraction completed! Total features: {len(features_df.columns)}")
print(final_df.head())

# Save the final dataset with features
final_df.to_csv(output_file, index=False)
print(f"\nFinal features dataset saved to: {output_file}")

print("\n=== Completed Successfully! ===")