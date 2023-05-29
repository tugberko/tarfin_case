import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def check_if_ever_overdraft(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks if an individual has ever delayed
    :return: This function returns nothing.
    """

    bill_amount_columns = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]

    dataframe["OVERDRAFT"] = (dataframe[bill_amount_columns].max(axis=1) / dataframe["LIMIT_BAL"] > 1).astype(int)

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = check_if_ever_overdraft(df)



# Correlation
print(df.corr()["DEFAULT"]["OVERDRAFT"])

# Descriptive stuff
print(df["OVERDRAFT"].describe())

# Defaulting ratios
age_counts = df.groupby('OVERDRAFT')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Overdraft Before")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Overdraft History")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
