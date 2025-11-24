import pandas as pd
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def check_preprocessed_data(train_file, test_file):
    """
    Check and report the quality of preprocessed train and test datasets,
    and save the report to check.xlsx
    """
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Train or test file not found!")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # --- 1. Basic info ---
    report = []

    report.append(["Train shape", train_df.shape])
    report.append(["Test shape", test_df.shape])

    # --- 2. Missing values ---
    report.append(["--- Missing Values Train ---", ""])
    for col, val in train_df.isnull().sum().items():
        report.append([col, val])

    report.append(["--- Missing Values Test ---", ""])
    for col, val in test_df.isnull().sum().items():
        report.append([col, val])

    # --- 3. Data types ---
    report.append(["--- Train Data Types ---", ""])
    for col, val in train_df.dtypes.items():
        report.append([col, str(val)])

    report.append(["--- Test Data Types ---", ""])
    for col, val in test_df.dtypes.items():
        report.append([col, str(val)])

    # --- 4. Numeric columns ranges ---
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')

    report.append(["--- Numeric Columns Ranges ---", ""])
    for col in numeric_cols:
        train_min = train_df[col].min()
        train_max = train_df[col].max()
        test_min = test_df[col].min()
        test_max = test_df[col].max()
        report.append(
            [col, f"Train min={train_min:.3f}, max={train_max:.3f} | Test min={test_min:.3f}, max={test_max:.3f}"])

    # --- 5. Binary / Dummy columns ---
    report.append(["--- Binary / Dummy Columns ---", ""])
    for col in train_df.columns:
        if train_df[col].nunique() <= 2 and col != 'price':
            report.append([col, f"Train unique={train_df[col].unique()}, Test unique={test_df[col].unique()}"])

    # --- 6. Target column ---
    report.append(["--- Target Column 'price' ---", ""])
    if 'price' in train_df.columns:
        report.append(["Train unique", train_df['price'].unique()])
        if 'price' in test_df.columns:
            report.append(["Test unique", test_df['price'].unique()])
        else:
            report.append(["Test unique", "No target column in test dataset"])

    # --- 7. Save report to Excel ---
    report_df = pd.DataFrame(report, columns=["Column / Info", "Value"])
    output_path = os.path.join(os.path.dirname(train_file), "check.xlsx")
    report_df.to_excel(output_path, index=False)
    print(f"✅ Preprocessed data check saved to {output_path}")


if __name__ == "__main__":
    Tk().withdraw()  # اخفاء نافذة tkinter الرئيسية
    train_file = askopenfilename(title="Select train_preprocessed.csv", filetypes=[("CSV files", "*.csv")])
    test_file = askopenfilename(title="Select test_preprocessed.csv", filetypes=[("CSV files", "*.csv")])

    check_preprocessed_data(train_file, test_file)