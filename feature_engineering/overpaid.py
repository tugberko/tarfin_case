import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def check_if_ever_overpaid(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function measures the individuals activity by counting non zero bill statements
    :return: This function returns nothing.
    """

    bill_statement_columns = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]

    dataframe["OVERPAID"] = (dataframe[bill_statement_columns] < 0).any(axis=1).astype(int)

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = check_if_ever_overpaid(df)



# Correlation
print(df.corr()["DEFAULT"]["OVERPAID"])

# Descriptive stuff
print(df["OVERPAID"].describe())

# Defaulting ratios
age_counts = df.groupby('OVERPAID')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Overpayment Status")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Overpayment")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
