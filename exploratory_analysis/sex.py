import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

sex_counts = df.groupby('SEX')["default payment next month"].value_counts(normalize=False).unstack()

print(sex_counts)

print(df.columns)

sex_counts.plot(kind="bar", stacked=True)
plt.ylabel("Defaulting Ratio")
plt.xlabel("Sex")
plt.xticks(rotation=0)
plt.title("Defaulting Ratio vs. Sex")
plt.grid(axis="y")
plt.legend(title='Default', loc='upper right')

plt.tight_layout()
plt.show()
