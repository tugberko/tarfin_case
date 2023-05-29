import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

df["SEX"] = df["SEX"].replace(1, "M")
df["SEX"] = df["SEX"].replace(2, "F")

df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
df["EDUCATION"] = df["EDUCATION"].replace(1, "grad")
df["EDUCATION"] = df["EDUCATION"].replace(2, "uni")
df["EDUCATION"] = df["EDUCATION"].replace(3, "hs")
df["EDUCATION"] = df["EDUCATION"].replace(4, "other")

df["MARRIAGE"] = df["MARRIAGE"].replace([0], 3)
df["MARRIAGE"] = df["MARRIAGE"].replace([1], "married")
df["MARRIAGE"] = df["MARRIAGE"].replace([2], "single")
df["MARRIAGE"] = df["MARRIAGE"].replace([3], "divorced")


grouped_data = df.groupby(['SEX', 'MARRIAGE', "EDUCATION"])['default payment next month'].value_counts(normalize=True).unstack()

# Plotting the stacked bar chart
grouped_data.plot(kind='bar', stacked=True)
plt.xlabel('Sex, Marital Status, Education')
plt.ylabel('Proportion')
plt.title('Distribution of Target Variable by Demographic Info')
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.show()