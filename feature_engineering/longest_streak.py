import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMN = "DEFAULT"


def len_longest_nonpositive_subseq(lst):
    subsequences = []
    start_index = None

    for i, num in enumerate(lst):
        if num <= 0:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            end_index = i - 1
            subsequence = lst[start_index:end_index + 1]
            subsequences.append(subsequence)
            start_index = None

    if start_index is not None:
        subsequence = lst[start_index:]
        subsequences.append(subsequence)

    maxlen = 0

    for current_seq in subsequences:
        if len(current_seq) >= maxlen:
            maxlen = len(current_seq)

    return maxlen





def measure_max_streak(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks if an individual has ever delayed
    :return: This function returns nothing.
    """

    payment_status_columns = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    max_streaks = []

    for _, row in dataframe[payment_status_columns].iterrows():
        seq = row.values
        max_streaks.append(len_longest_nonpositive_subseq(seq))

    dataframe["LONGEST_STREAK"] = max_streaks

    return dataframe


df = pd.read_csv("/home/tugberkozdemir/Workspace/tarf/data/temp.csv")

df = measure_max_streak(df)

# Correlation
print(df.corr()["DEFAULT"]["LONGEST_STREAK"])

# Descriptive stuff
print(df["LONGEST_STREAK"].describe())

# Defaulting ratios
age_counts = df.groupby('LONGEST_STREAK')[DEFAULT_COLUMN].value_counts(normalize=True).unstack()

age_counts.plot(kind="bar", stacked=True)
plt.xlabel("Longest Streak (months)")
plt.ylabel("Defaulting Ratio")
plt.title("Defaulting Ratio vs. Longest Streak")
plt.xticks(ticks=plt.gca().get_xticks(), labels=['{:.2f}'.format(x) for x in plt.gca().get_xticks()])
plt.grid(axis="y")
plt.legend(title='Defaulting Status', loc='upper right')
plt.tight_layout()
plt.show()
