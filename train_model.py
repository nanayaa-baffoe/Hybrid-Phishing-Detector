import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("=== Section 4: Model Training & Evaluation ===\n")

# Load the features dataset from Section 3
data_dir = Path("Dataset")
input_file = data_dir / "final_features_dataset.csv"

if not input_file.exists():
    print("Error: final_features_dataset.csv not found! Please run feature_engineering.py first.")
    exit()

df = pd.read_csv(input_file)

# Use only the numerical features (exclude 'text' column)
feature_columns = [col for col in df.columns if col not in ['text', 'label']]
X = df[feature_columns]
y = df['label']

print(f"Training on {X.shape[1]} features with {len(df)} samples.")

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples\n")

# ===================== TRAIN MULTIPLE MODELS =====================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("Training models... Please wait.\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {"Accuracy": acc, "F1 Score": f1}
    
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}\n")

# Show comparison table
results_df = pd.DataFrame(results).T
print("=== MODEL COMPARISON ===")
print(results_df.round(4))

# Save the best model (Random Forest)
best_model = models["Random Forest"]
best_model.fit(X_train, y_train)   # Train on full training data

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(best_model, models_dir / "best_ml_model.pkl")

print(f"\nBest model (Random Forest) saved to: models/best_ml_model.pkl")
print("\n=== Section 4 Completed Successfully! ===")