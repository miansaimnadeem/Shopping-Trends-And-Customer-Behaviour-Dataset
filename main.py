import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Current working directory:", os.getcwd())
try:
    df = pd.read_csv(r'F:\Universty\4th Semester\Intro to Ds\Shopping Trends And Customer Behaviour Dataset.csv')
except FileNotFoundError:
    print("Error: The file 'Shopping Trends And Customer Behaviour Dataset.csv' was not found in the specified path.")
    exit()
    
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

categorical_features=['Gender', 'Category', 'location', 'Payment Method', 'Frequency of Purchases']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)
    plt.title(f'Distribution of {feature}')
    plt.xlabel('Count')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f'{feature}_distribution.png')
    plt.close()
    
    
numerical_features=['Age', 'Purchase Amount (USD)', 'Review Rating']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{feature}_distribution.png')
    plt.close()
    
    
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Purchase Amount (USD)', data=df)
plt.title('Purchase Amount by Category')
plt.xlabel('Category')
plt.tight_layout()
plt.savefig('Purchase_Amount_by_Category_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Purchase Amount (USD)', data=df)
plt.title('Purchase Amount (USD) by Gender')
plt.xlabel('Gender')
plt.ylabel('Purchase Amount (USD)')
plt.tight_layout()
plt.savefig('purchase_amount_by_gender.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=df)
plt.title('Age vs. Purchase Amount (USD)')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.tight_layout()
plt.savefig('age_vs_purchase_amount.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

bins = [10, 20, 30, 40, 50, 60, 70]
labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


df['Spending Category'] = pd.qcut(df['Purchase Amount (USD)'], q=4, labels=['Low Spender', 'Medium Spender', 'High Spender', 'Very High Spender'])

print("\nDataFrame with new features:")
print(df.head())

df.to_csv('engineered_shopping_trends.csv', index=False)
