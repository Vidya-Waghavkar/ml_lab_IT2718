import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

     

data = {
    'Age': [56, 46, 32, 25, 38, 56, 36, 40, 28, 28, 41, 53],
    'Salary': [31023, 56090, 82221, 1000000, 15769, 74735, 79925, 20311, 118355, 199779, 100305, 174765],
    'YearsExperience': [11, 16, 9, 15, 14, 14, 18, 11, 19, 2, 4, 18],
    'Purchases': [134, 20, 328, 166, 273, 387, 88, 315, 13, 241, 264, 345]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
     

scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)
df_decimal = df / (10 ** np.ceil(np.log10(df.abs().max())))
df_log = np.log1p(df)
scaler_unit = Normalizer(norm='l2')
df_unit = pd.DataFrame(scaler_unit.fit_transform(df), columns=df.columns)

     

save_path = "/mnt/data/normalization_demo"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/original_data.csv", index=False)
df_minmax.to_csv(f"{save_path}/minmax_normalized.csv", index=False)
df_std.to_csv(f"{save_path}/zscore_standardized.csv", index=False)
df_decimal.to_csv(f"{save_path}/decimal_scaled.csv", index=False)
df_log.to_csv(f"{save_path}/log_transformed.csv", index=False)
df_unit.to_csv(f"{save_path}/unit_vector_normalized.csv", index=False)

     

print("\n🔹 Min-Max Normalized:\n", df_minmax.head())
print("\n🔹 Z-score Standardized:\n", df_std.head())
print("\n🔹 Decimal Scaled:\n", df_decimal.head())
print("\n🔹 Log Transformed:\n", df_log.head())
print("\n🔹 Unit Vector Normalized:\n", df_unit.head())

     

print("\n Download links (click if supported):")
print(f"[Original CSV]({save_path}/original_data.csv)")
print(f"[Min-Max CSV]({save_path}/minmax_normalized.csv)")
print(f"[Z-score CSV]({save_path}/zscore_standardized.csv)")
print(f"[Decimal Scaling CSV]({save_path}/decimal_scaled.csv)")
print(f"[Log Transform CSV]({save_path}/log_transformed.csv)")
print(f"[Unit Vector CSV]({save_path}/unit_vector_normalized.csv)")

     

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Salary'], bins=10, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title("Original Salary Distribution")
sns.histplot(df_minmax['Salary'], bins=10, kde=True, ax=axes[1], color='orange')
axes[1].set_title("Min-Max Normalized Salary")
plt.tight_layout()
plt.show()

     

plt.figure(figsize=(8,5))
sns.kdeplot(df['Salary'], label='Original', color='blue')
sns.kdeplot(df_minmax['Salary'], label='Min-Max Normalized', color='orange')
plt.title("Comparison: Original vs Min-Max Normalized Salary")
plt.xlabel("Salary Value")
plt.legend()
plt.show()

     

plt.figure(figsize=(10,6))
sns.kdeplot(df['Salary'], label='Original')
sns.kdeplot(df_std['Salary'], label='Z-score Standardized')
sns.kdeplot(df_log['Salary'], label='Log Transformed')
sns.kdeplot(df_decimal['Salary'], label='Decimal Scaled')
sns.kdeplot(df_unit['Salary'], label='Unit Vector Normalized')
sns.kdeplot(df_minmax['Salary'], label='Min-Max Normalized')
plt.title("Salary Distribution Across Different Normalization Techniques")
plt.xlabel("Transformed Salary Value")
plt.legend()
plt.show()