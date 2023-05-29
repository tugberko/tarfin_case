import matplotlib.pyplot as plt
import pandas as pd

from config import PATH_TO_RAW_DATA

df = pd.read_excel(PATH_TO_RAW_DATA, header=[1])

counts = df["default payment next month"].value_counts()

plt.pie(counts, labels=counts.index, autopct='%1.1f%%')

plt.title('Target Variable Distribution')
plt.legend(title='Default', loc='upper right')
plt.tight_layout()
plt.show()
