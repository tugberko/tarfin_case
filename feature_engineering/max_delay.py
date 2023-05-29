import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def measure_max_delay(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks if an individual has ever delayed
    :return: This function returns nothing.
    """

    payment_status_columns = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    dataframe["MAX_DELAY"] = (dataframe[payment_status_columns]).max(axis=1)

    dataframe["MAX_DELAY"] = dataframe["MAX_DELAY"].clip(lower=0)

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = measure_max_delay(df)



# Correlation
print(df.corr()["DEFAULT"]["MAX_DELAY"])

# Descriptive stuff
print(df["MAX_DELAY"].describe())

# Defaulting ratios
age_counts = df.groupby('MAX_DELAY')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Max. Delay (months)")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Max. Delay")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
