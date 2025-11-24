import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def preprocess_data():
    # --- 1. Ask user to select files ---
    Tk().withdraw()  # Hide main window

    print("Please select the TRAIN file (train.csv)...")
    train_path = askopenfilename(title="Select train.csv", filetypes=[("CSV files", "*.csv")])
    if not train_path:
        raise FileNotFoundError("No train file selected!")

    print("Please select the TEST file (test.csv)...")
    test_path = askopenfilename(title="Select test.csv", filetypes=[("CSV files", "*.csv")])
    if not test_path:
        raise FileNotFoundError("No test file selected!")

    print(f"Using train file: {train_path}")
    print(f"Using test file: {test_path}")

    # --- 2. Load Data ---
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Data loaded successfully!")

    # --- 3. Add flag and combine ---
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # --- 4. Binary Mapping ---
    binary_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support']
    binary_map = {'Yes': 1, 'No': 0}
    for col in binary_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].map(binary_map)

    # Target Mapping
    if 'price' in full_df.columns:
        full_df['price'] = full_df['price'].map({'expensive': 1, 'non-expensive': 0})

    # --- 5. Clean specific columns safely ---
    text_cols = ['brand', 'Processor_Series', 'os_version']
    for col in text_cols:
        if col in full_df.columns:
            # فقط تحويل النصوص لupper، وترك القيم الغير نصية كما هي
            full_df[col] = full_df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)

    if 'memory_card_size' in full_df.columns:
        def clean_memory_card_size(val):
            if pd.isna(val):
                return 0
            val = str(val)
            if 'TB' in val:
                return float(val.replace(' TB', '')) * 1024
            elif 'GB' in val:
                return float(val.replace(' GB', ''))
            return 0
        full_df['memory_card_size'] = full_df['memory_card_size'].apply(clean_memory_card_size)

    # --- 6. Encoding ---
    for col in text_cols:
        if col in full_df.columns:
            le = LabelEncoder()
            full_df[col] = le.fit_transform(full_df[col].astype(str))

    ohe_cols = ['Processor_Brand', 'Performance_Tier', 'RAM Tier', 'Notch_Type', 'os_name']
    exist_ohe_cols = [c for c in ohe_cols if c in full_df.columns]
    full_df = pd.get_dummies(full_df, columns=exist_ohe_cols, drop_first=True)

    # --- 7. Scaling ---
    exclude_cols = ['price', 'is_train']
    numeric_cols = full_df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    train_preprocessed = full_df[full_df['is_train'] == 1].drop(['is_train'], axis=1)
    test_preprocessed = full_df[full_df['is_train'] == 0].drop(['is_train'], axis=1)

    scaler = StandardScaler()
    train_preprocessed[numeric_cols] = scaler.fit_transform(train_preprocessed[numeric_cols])
    test_preprocessed[numeric_cols] = scaler.transform(test_preprocessed[numeric_cols])

    # --- 8. Save Output ---
    output_dir = os.path.dirname(train_path)
    train_preprocessed.to_csv(os.path.join(output_dir, 'train_preprocessed.csv'), index=False)
    test_preprocessed.to_csv(os.path.join(output_dir, 'test_preprocessed.csv'), index=False)

    print(f"Preprocessing complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    preprocess_data()