import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ackley.csv")
# Assuming df is your DataFrame
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10))
plt.show()