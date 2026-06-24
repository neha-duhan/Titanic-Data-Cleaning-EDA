import pandas as pd
import numpy as np
import kagglehub
import os

# 1. Download and Load Data
 
path = kagglehub.dataset_download(
    "akshaysehgal/titanic-data-for-data-preprocessing"
)

print("Dataset Path:", path)
print("Files in Dataset Folder:", os.listdir(path))

# Load CSV file
csv_file = os.path.join(path, "titanic_data.csv")
df = pd.read_csv(csv_file)

# 2. Display Dataset
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

# 3. Find Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 4. Fill Missing Numerical Values with Mean
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Remove Duplicate Rows
duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before Removal:", duplicates_before)

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()
print("Duplicate Rows After Removal:", duplicates_after)

# 6. Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# 7. Basic Exploratory Analysis

print("\nAverage Age:", df["age"].mean())
print("Highest Fare:", df["fare"].max())
print("Lowest Fare:", df["fare"].min())
print("Average Fare:", df["fare"].mean())
print("Total Passengers:", len(df))
print("\nSurvival Count:")
print(df["survived"].value_counts())

# 8. Findings
print("\n----- FINDINGS -----")
print("1. Dataset loaded successfully from Kaggle.")
print("2. Missing numerical values were replaced with the column mean.")
print("3. Duplicate rows were removed.")
print("4. Statistical summary was generated using describe().")
print(f"5. Average Age of passengers: {df['age'].mean():.2f}")
print(f"6. Highest Fare paid: {df['fare'].max():.2f}")
print(f"7. Total Passengers after cleaning: {len(df)}")