import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Processed Data
df = pd.read_csv('C:\\Users\\Youssef Elshemy\\Documents\\Projects\\Smartphone-Price-Prediction\\data\\processed\\processed_train.csv')

# Calculate Correlations with Price: tells us how much each feature pushes the price to be "1" (Expensive)
correlations = df.corr()['price'].sort_values()
correlations = correlations.drop('price')
correlations = correlations.dropna()

# Get Top 10 Positive (Push Price UP) and Negative (Push Price DOWN)
top_pos = correlations.tail(10)
top_neg = correlations.head(10)

print("Top Features for Expensive Phones:\n", top_pos)
print("\nTop Features for Budget Phones:\n", top_neg)

# Visualization: Feature Drivers (Bar Chart)
plt.figure(figsize=(12, 8))
top_features = pd.concat([top_neg, top_pos])

sns.barplot(
    x=top_features.values,
    y=top_features.index,
    hue=top_features.index,
    palette='RdBu_r',
    legend=False
)

plt.title('What Makes a Phone Expensive? (Correlation with Price)')
plt.xlabel('Correlation Coefficient (Left=Cheaper, Right=Expensive)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()