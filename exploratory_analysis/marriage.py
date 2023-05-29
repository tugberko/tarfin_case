import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

marriage_counts = df.groupby('MARRIAGE')["default payment next month"].value_counts(normalize=True).unstack()

marriage_counts.plot(kind="bar", stacked=True)
plt.ylabel("Defaulting Ratio")
plt.xlabel("Marital Status")
plt.xticks(rotation=0)
plt.title("Defaulting Ratio vs. Marital Status")
plt.grid(axis="y")
plt.legend(title='Default', loc='upper right')

plt.tight_layout()
plt.show()

