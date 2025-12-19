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

# Visualization 1: Feature Drivers (Bar Chart)
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

# Visualization 2: Scaled Distributions (Boxplots)
# We look at the top 3 continuous features: Clock Speed, RAM, and Storage
# Since data is scaled 0-1, we can easily compare them on the same plot
plot_cols = ['Clock_Speed_GHz', 'RAM Size GB', 'Storage Size GB']

plt.figure(figsize=(14, 6))
# Melt data to long format for seaborn
melted_df = df.melt(id_vars='price', value_vars=plot_cols, var_name='Feature', value_name='Scaled Value')

sns.boxplot(x='Feature', y='Scaled Value', hue='price', data=melted_df, palette='Set2')
plt.title('Distribution of Key Specs: Budget (0) vs Expensive (1)')
plt.ylabel('Scaled Value (0=Min, 1=Max)')
plt.legend(title='Price Class', loc='upper left', labels=['Non-Expensive', 'Expensive'])
plt.tight_layout()
plt.show()