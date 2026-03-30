import pandas as pd
import random
import json
import time
import joblib
import numpy as np

# Updated paths based on your provided structure
DATA_DIR = '../data/processed/'
MODELS_DIR = '../models/'

def get_inference_sample():
    """
    Loads processed training data and artifacts to generate 
    a JSON sample formatted for inference.
    """
    try:
        # 1. Load engineering artifacts
        feature_cols = joblib.load(MODELS_DIR + 'feature_cols.pkl')
        encoders = joblib.load(MODELS_DIR + 'label_encoders.pkl')
        imputer = joblib.load(MODELS_DIR + 'imputer.pkl')
        # Stats are available if you need to normalize 'amount' later
        stats = joblib.load(MODELS_DIR + 'amount_stats.pkl') 
        
        # 2. Load the Train Set (Features and Target)
        # Using Parquet as per your saved files
        X_train = pd.read_parquet(DATA_DIR + 'X_tr_raw.parquet')
        y_train = pd.read_parquet(DATA_DIR + 'y_tr.parquet')
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 3. Pick a random index from the training set
    random_idx = random.randint(0, len(X_train) - 1)
    df_sample = X_train.iloc[[random_idx]].copy()
    
    # Extract metadata/target
    # Assumes y_tr.parquet has a column named 'isFraud'
    target_val = int(y_train.iloc[random_idx].values[0])
    
    # Use TransactionAmt if it exists in your raw features, else default
    raw_amount = float(df_sample.get('TransactionAmt', [0.0]).iloc[0])

    # 4. PRE-PROCESSING
    # Ensure we only have the columns the model expects
    df_features = df_sample[feature_cols].copy()

    # Apply Label Encoding
    for col, le in encoders.items():
        if col in df_features.columns:
            val = str(df_features[col].iloc[0])
            try:
                # transform expects a list/array
                df_features[col] = le.transform([val])[0]
            except (ValueError, KeyError):
                # Handle unseen labels (common in inference)
                df_features[col] = 0 

    # Apply Imputation
    features_imputed = imputer.transform(df_features)
    
    # Map back to dictionary
    processed_dict = dict(zip(feature_cols, features_imputed[0]))
    '''
    {
    "card_id": "card_12345",
    "amount": 150.00,
    "merchant_id": "merch_999",
    "timestamp": 1711532400.0,
    "ip_address": "192.168.1.1",
    "account_age_hours": 48.0,
    "bin_number": "512345",
    "features": {}
    }
    '''
    # 5. Construct Final JSON Structure
    # This structure mimics a real-time API payload
    output = {
        "isFraud_actual": target_val, # For validation/testing purposes
        "timestamp": float(time.time()),
        "card_id": f"card_{random.randint(1000, 9999)}",
        "amount": raw_amount, 
        "merchant_id": f"merch_{random.randint(100, 999)}",
        "account_age_hours": float(random.randint(10, 400)),
        "ip_address": "192.168.1.1",
        "bin_number": str(random.randint(100000, 999999)),
        "features": {k: (v.item() if hasattr(v, 'item') else v) for k, v in processed_dict.items()}
    }

    # 6. Save to JSON
    output_filename = DATA_DIR + f'inference_sample_{target_val}.json'
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Success! Inference sample saved to: {output_filename}")
    print(f"Sample Index: {random_idx} | Actual Label: {target_val}")

if __name__ == "__main__":
    get_inference_sample()