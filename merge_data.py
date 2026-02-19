import pandas as pd
import requests
import io
import os

def load_enron():
    print("Loading Enron dataset from local file...")
    csv_path = "enron_emails.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please download it first.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        # Enron has 'text' and 'spam' (1/0)
        # Rename columns
        if 'spam' in df.columns:
            df = df.rename(columns={'spam': 'label'})
        
        # Map 1->spam, 0->ham if 'label' is numeric
        if df['label'].dtype in ['int64', 'float64']:
             df['label'] = df['label'].map({1: 'spam', 0: 'ham'})
        
        # Ensure text column is string
        df['text'] = df['text'].astype(str)
        
        print(f"Enron data loaded: {len(df)} rows")
        return df[['label', 'text']]
    except Exception as e:
        print(f"Error loading enron data: {e}")
        return pd.DataFrame()

def load_existing():
    if not os.path.exists("dataset.csv"):
        return pd.DataFrame()
    
    print("Loading existing dataset...")
    try:
        # Try reading with v1, v2 headers
        df = pd.read_csv("dataset.csv", encoding="latin-1")
        if 'v1' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        elif 'spam' in df.columns: # fallback if already processed
             df['label'] = df['spam'].map({1: 'spam', 0: 'ham'})
        
        df = df[['label', 'text']]
        print(f"Existing data loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return pd.DataFrame()

# Main execution
df_enron = load_enron()
df_old = load_existing()

# Combine
print("Merging datasets...")
df_final = pd.concat([df_old, df_enron], ignore_index=True)

# Remove duplicates
len_before = len(df_final)
df_final.drop_duplicates(subset=['text'], inplace=True)
print(f"Removed {len_before - len(df_final)} duplicates")

# Shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
print(f"Saving final dataset with {len(df_final)} rows...")
df_final.to_csv("dataset.csv", index=False)
print("Done!")
