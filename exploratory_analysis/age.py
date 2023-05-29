import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

DEFAULT_COLUMN = "default payment next month"

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

# AGE DISTRIBUTION

default_data = df[df[DEFAULT_COLUMN] == 1]['AGE']
non_default_data = df[df[DEFAULT_COLUMN] == 0]['AGE']

plt.hist([default_data, non_default_data], bins=20, edgecolor='black', stacked=True, label=['Default', 'Non-Default'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution - Default vs. Non-Default')
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()

# DEFAULTING RATIOS2

df['AGE'] = pd.cut(df['AGE'], bins=10, precision=0)

age_counts = df.groupby('AGE')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

print(df.columns)

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Age Group")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Age Group")
plt.xticks(rotation=90)
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
