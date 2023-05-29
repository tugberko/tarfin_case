import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from config import PATH_TO_RAW_DATA

DEFAULT_COLUMN = "default payment next month"

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

bins = np.arange(0, 1000000, 50000)  # Adjust the bin range and width as needed

# Update the x-axis ticks to align with the bin edges
plt.xticks(bins)

# LIMIT/BALANCE

default_data = df[df[DEFAULT_COLUMN] == 1]['LIMIT_BAL']
non_default_data = df[df[DEFAULT_COLUMN] == 0]['LIMIT_BAL']

plt.hist([default_data, non_default_data], bins=bins, edgecolor='black', stacked=True, label=['Default', 'Non-Default'])
plt.xlabel('Limit/Balance')
plt.ylabel('Count')
plt.title('Limit/Balance Distribution - Default vs. Non-Default')
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='x')


def format_ticks(x, pos):
    if x >= 1000:
        value = '{:.0f}k'.format(x * 1e-3)
    else:
        value = '{:.0f}'.format(x)
    return value


# Set the custom tick formatter
formatter = ticker.FuncFormatter(format_ticks)
plt.gca().xaxis.set_major_formatter(formatter)

plt.show()

# DEFAULTING RATIOS

age_counts = df.groupby('LIMIT_BAL')[DEFAULT_COLUMN]

print(age_counts.value_counts(normalize=True).unstack())

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Limit/Balance Group")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Limit/Balance Group")
plt.xticks(rotation=90)
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
