import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# Load Dataset
df = pd.read_csv("Supermarket_Sales.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Info")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Sales by Branch
plt.figure(figsize=(6,4))
sns.countplot(x="Branch", data=df)
plt.title("Sales by Branch")
plt.show()

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Customer Gender")
plt.show()

# Product Line
plt.figure(figsize=(8,5))
sns.countplot(y="Product Line", data=df)
plt.title("Product Line Distribution")
plt.show()

# Payment Methods
plt.figure(figsize=(6,4))
sns.countplot(x="Payment", data=df)
plt.title("Payment Methods")
plt.show()

# Total Sales
plt.figure(figsize=(7,4))
sns.histplot(df["Total"], bins=20, kde=True)
plt.title("Total Sales Distribution")
plt.show()

# Ratings
plt.figure(figsize=(7,4))
sns.histplot(df["Rating"], bins=15, kde=True)
plt.title("Customer Ratings")
plt.show()

# Correlation
plt.figure(figsize=(8,6))
corr=df[["Unit Price","Quantity","COGS","Gross Income","Rating","Total"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[["Unit Price","Quantity","Total","Rating"]])
plt.show()

print("\nBranch-wise Total Sales")
print(df.groupby("Branch")["Total"].sum())

print("\nCity-wise Total Sales")
print(df.groupby("City")["Total"].sum())

print("\nTop Product Lines")
print(df.groupby("Product Line")["Total"].sum().sort_values(ascending=False))

df.to_csv("Cleaned_Supermarket_Sales.csv", index=False)

print("\nEDA Completed Successfully!")
