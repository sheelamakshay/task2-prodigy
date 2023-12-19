import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

np.random.seed(42)

feature_1 = np.random.rand(150)
feature_2 = np.random.choice(['1', '2', '3', 'B'], size=150)
feature_3 = np.random.choice(['A', 'B', 'C'], size=150)
target = np.random.choice([0, 1, 2], size=150)

example_data = pd.DataFrame({
    'sepal length (cm)': feature_1,
    'sepal width (cm)': feature_2,
    'petal length (cm)': feature_3,
    'target': target
})

data = example_data.copy()

column_to_convert = 'sepal width (cm)'
data[column_to_convert] = pd.to_numeric(data[column_to_convert], errors='coerce')

print(data.isnull().sum())
print(data.describe())

numeric_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_columns].corr()
print(correlation_matrix)

sns.pairplot(data, hue='target', height=2.5)
plt.show()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(data.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target', y=feature, data=data)
    plt.title(f'Boxplot of {feature} by Species')
plt.tight_layout()
plt.show()
