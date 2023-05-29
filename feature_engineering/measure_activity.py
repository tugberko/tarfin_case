import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def measure_activity(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function measures the individuals activity by counting non zero bill statements
    :return: This function returns nothing.
    """

    payment_status_columns = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    dataframe["ACTIVITY"] = 1 - (dataframe[payment_status_columns] == -2).sum(axis=1) / 6

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = measure_activity(df)

# Correlation
print(df.corr()["DEFAULT"]["ACTIVITY"])

# Descriptive stuff
print(df["ACTIVITY"].describe())

# Defaulting ratios
age_counts = df.groupby('ACTIVITY')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Activity (months)")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Card Activity")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
