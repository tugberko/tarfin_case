import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PATH_TO_RAW_DATA

DEFAULT_COLUMN = "default payment next month"

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])
# Get rid of irrelevant data
df.drop(columns=["ID"], inplace=True)

# Calculation of the correlation matrix
correlation_matrix = df.corr()

# No need to display same information twice and we know self correlations are 1
# so, get rid of upper triangular part.
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot
plt.figure(figsize=(12, 12))  # Adjust the figure size as desired
sns.heatmap(correlation_matrix, cmap='viridis', annot=True, mask=mask, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
