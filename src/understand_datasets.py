import pandas as pd

# Load the dataset
train_df = pd.read_csv('data\\raw\\train.csv')
test_df = pd.read_csv('data\\raw\\test.csv')

# Basic info
print("🔹 Train Dataset Info:")
print(train_df.info())
print("\n🔹 Test Dataset Info:")
print(test_df.info())
print("\n")

# First few rows
print("🔹 Train: First 5 Rows:")
print(train_df.head())
print("\n🔹 Test: First 5 Rows:")
print(test_df.head())
print("\n")

# Summary statistics
print("🔹 Train: Summary Statistics:")
print(train_df.describe(include="all"))
print("\n🔹 Test: Summary Statistics:")
print(test_df.describe(include="all"))
print("\n")

# Missing values
print("🔹 Train: Missing Values:")
print(train_df.isnull().sum())
print("\n🔹 Test: Missing Values:")
print(test_df.isnull().sum())
print("\n")

# Duplicates
print("🔹 Train: Duplicates:")
print(train_df.duplicated().sum())
print("\n🔹 Test: Duplicates:")
print(test_df.duplicated().sum())
print("\n")

# Unique values per column
print("🔹 Train: Unique Values in Each Column:")
for col in train_df.columns:
   print(f"{col}: {train_df[col].nunique()} unique values")
print("\n🔹 Test: Unique Values in Each Column:")
for col in test_df.columns:
   print(f"{col}: {test_df[col].nunique()} unique values")
print("\n")