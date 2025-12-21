import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_preprocessed_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    # Load Data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop duplicates
    train = train.drop_duplicates()

    # Combine for consistent encoding
    train['is_train'] = 1
    test['is_train'] = 0
    full_df = pd.concat([train, test], axis=0).reset_index(drop=True)

    # --- CLEANING ---

    # Fix Brands
    full_df['brand'] = full_df['brand'].str.title()

    # Parse Storage (e.g., "1 TB" -> 1024)
    def parse_storage(size_str):
        if pd.isna(size_str): return 0
        s = str(size_str).upper()
        if 'TB' in s: return float(s.replace('TB', '').strip()) * 1024
        if 'GB' in s: return float(s.replace('GB', '').strip())
        try:
            return float(s)
        except:
            return 0

    full_df['memory_card_size'] = full_df['memory_card_size'].apply(parse_storage)

    # 3. Parse OS Version (e.g., "v12" -> 12.0)
    def parse_version(v):
        try:
            match = re.search(r'(\d+(\.\d+)?)', str(v))
            if match: return float(match.group(1))
            return 0.0
        except:
            return 0.0

    full_df['os_version'] = full_df['os_version'].apply(parse_version)

    # --- ENCODING ---

    # Map 'price' (Target)
    full_df['price'] = full_df['price'].map({'expensive': 1, 'non-expensive': 0})

    # Map Yes/No columns
    binary_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support']
    for col in binary_cols:
        full_df[col] = full_df[col].map({'Yes': 1, 'No': 0})

    # Ordinal Encoding (Preserving Rank)
    tier_map = {'Unknown': 0, 'Budget': 1, 'Mid-Range': 2, 'High-End': 3, 'Flagship': 4}
    full_df['Performance_Tier'] = full_df['Performance_Tier'].map(tier_map)
    full_df['RAM Tier'] = full_df['RAM Tier'].map(tier_map)

    # One-Hot Encoding (Nominal Data)
    # We drop 'Processor_Series' to reduce noise
    cols_to_dummy = ['brand', 'Processor_Brand', 'os_name', 'Notch_Type']
    full_df = pd.get_dummies(full_df, columns=cols_to_dummy, drop_first=True)
    full_df = full_df.drop('Processor_Series', axis=1)

    # --- SPLITTING & SCALING ---

    # Separate Train and Test
    train_processed = full_df[full_df['is_train'] == 1].drop('is_train', axis=1)
    test_processed = full_df[full_df['is_train'] == 0].drop('is_train', axis=1)

    # Split Train into Train (80%) and Validation (20%)
    # Stratify ensures the proportion of expensive/non-expensive phones is consistent
    x_train_full = train_processed.drop('price', axis=1)
    y_train_full = train_processed['price']

    features_train, features_validation, target_train, target_validation = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    features_test = test_processed.drop('price', axis=1)
    target_test = test_processed['price']

    # Scale Features (Using MinMaxScaler for Naive Bayes compatibility)
    scaler = MinMaxScaler()

    # Fit ONLY on training data to avoid leakage
    scaler.fit(features_train)

    # Transform all
    features_train = pd.DataFrame(scaler.transform(features_train), columns=features_train.columns)
    features_validation = pd.DataFrame(scaler.transform(features_validation), columns=features_validation.columns)
    features_test = pd.DataFrame(scaler.transform(features_test), columns=features_test.columns)

    # Save
    train_processed.to_csv('data\\processed\\processed_train.csv', index=False)
    test_processed.to_csv('data\\processed\\processed_test.csv', index=False)

    return features_train, target_train, features_validation, target_validation, features_test, target_test