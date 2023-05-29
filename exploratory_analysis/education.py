import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

edu_counts = df.groupby('EDUCATION')["default payment next month"].value_counts(normalize=True).unstack()

edu_counts.plot(kind="bar", stacked=True)
plt.ylabel("Defaulting Ratio")
plt.xlabel("Education")
plt.xticks(rotation=0)
plt.title("Defaulting Ratio vs. Education")
plt.grid(axis="y")
plt.legend(title='Default', loc='upper right')

plt.tight_layout()
plt.show()

