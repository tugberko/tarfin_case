import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def check_if_ever_delayed(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks if an individual has ever delayed
    :return: This function returns nothing.
    """

    payment_status_columns = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    dataframe["DELAYED"] = (dataframe[payment_status_columns] > 0).any(axis=1).astype(int)

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = check_if_ever_delayed(df)



# Correlation
print(df.corr()["DEFAULT"]["DELAYED"])

# Descriptive stuff
print(df["DELAYED"].describe())

# Defaulting ratios
age_counts = df.groupby('DELAYED')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Delayed Before")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Delay History")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
