# ----------------------------------------------------------
# 🧩 Titanic Dataset - Exploratory Data Analysis (EDA)
# ----------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    df = sns.load_dataset("titanic")
except:
    df = pd.read_csv("titanic.csv")

# Basic info
print("----- DATA INFO -----")
print(df.info())
print("\n----- DESCRIPTION -----")
print(df.describe())
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Value counts
print("\n----- VALUE COUNTS -----")
print("Sex:\n", df['sex'].value_counts())
print("Class:\n", df['class'].value_counts())
print("Embarked:\n", df['embarked'].value_counts())

# Visualizations
sns.set_style("whitegrid")

# Age distribution
plt.figure(figsize=(7,5))
sns.histplot(df['age'], bins=20, kde=True, color='teal')
plt.title("Age Distribution of Passengers")
plt.show()

# Fare by class
plt.figure(figsize=(7,5))
sns.boxplot(x='class', y='fare', data=df, palette='Set2')
plt.title("Fare Distribution by Class")
plt.show()

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df, palette='viridis')
plt.title("Survival Count")
plt.xlabel("0 = Not Survived | 1 = Survived")
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df, palette='coolwarm')
plt.title("Survival by Gender")
plt.show()

# Class vs Survival
plt.figure(figsize=(7,5))
sns.countplot(x='class', hue='survived', data=df, palette='Spectral')
plt.title("Survival by Passenger Class")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[['age','fare','survived','pclass']], hue='survived', palette='husl')
plt.show()

# Summary
print("""
--------------------------------------------
📋 SUMMARY OF INSIGHTS
--------------------------------------------
1. Average passenger age ≈ 29 years.
2. About 38% of passengers survived.
3. Women and first-class passengers had the highest survival rates.
4. Fare and class are positively correlated.
5. Most passengers embarked from Southampton.
6. Missing values exist in 'age', 'embarked', and 'deck' columns.
--------------------------------------------
""")