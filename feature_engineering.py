from typing import List

import pandas as pd

from config import PATH_TO_TEMPORARY_DATA


class FeatureEngineeringStage:

    def __init__(self):
        self.data = pd.DataFrame

        self.bill_statement_columns: List[str] = []
        self.payment_amount_columns: List[str] = []
        self.payment_status_columns: List[str] = []

    def prepare_column_lists(self):
        self.bill_statement_columns = [column for column in self.data.columns if column.startswith("BILL_AMT")]

        self.payment_amount_columns = [column for column in self.data.columns if column.startswith("PAY_AMT")]

        self.payment_status_columns = [f"PAY_{i}" for i in [1, 2, 3, 4, 5, 6]]

    def load_data(self):
        """
        This function reads the data produced by previous stage.
        :return: This function returns nothing.
        """

        try:
            print("Loading data...")
            self.data = pd.read_csv(PATH_TO_TEMPORARY_DATA)
            print("Data has been loaded.")
        except FileNotFoundError:
            print(f"No such file {PATH_TO_TEMPORARY_DATA}")
            quit()

    def measure_activity(self):
        """
        This function measures the individuals activity by counting non zero bill statements
        :return: This function returns nothing.
        """

        self.data["ACTIVITY"] = 1 - (self.data[self.payment_status_columns] == -2).sum(axis=1) / 6

    def check_if_ever_overpaid(self):
        """
        This function checks if an individual has ever overpaid (negative bill amount)
        :return: This function returns nothing
        """

        self.data["OVERPAID"] = (self.data[self.bill_statement_columns] < 0).any(axis=1).astype(int)

    def check_if_ever_delayed(self):
        """
        This function checks if an individual has ever delayed paying.
        :return: This function returns nothing.
        """

        self.data["DELAYED"] = (self.data[self.payment_status_columns] > 0).any(axis=1).astype(int)

    def check_if_ever_overdraft(self):
        """
        This function checks if an individual has ever overdraft
        :return: This function returns nothing.
        """

        self.data["OVERDRAFT"] = (self.data[self.bill_statement_columns] > 0).any(axis=1).astype(int)

    def measure_maximum_delay(self):
        """
        This function measures the highest delay individual achieved.
        :return: This function returns nothing.
        """

        self.data["MAX_DELAY"] = (self.data[self.payment_status_columns]).max(axis=1).clip(lower=0)

    @staticmethod
    def len_longest_nonpositive_subseq(lst):
        """
        This utility function finds the length of longest non positive subsequence within a list.
        :param lst:
        :return: Length of the longest non positive subsequence
        """
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

    def measure_longest_streak(self):
        """
        This function measures the length of the widest time window where individual never delayed.
        :return: This function returns nothing.
        """

        max_streaks = []

        for _, row in self.data[self.payment_status_columns].iterrows():
            seq = row.values
            max_streaks.append(self.len_longest_nonpositive_subseq(seq))

        self.data["LONGEST_STREAK"] = max_streaks


    def dump_csv(self):
        """
        This function dumps the data for oncoming stages
        :return: This function returns nothing.
        """
        print("Dumping...")

        self.data.to_csv(PATH_TO_TEMPORARY_DATA, index=False)

        print("Dumped.")

    def execute_stage(self):
        """
        This function executes the current stage.
        :return: This function returns nothing.
        """

        print("Beginning stage: Feature Engineering")

        self.load_data()

        self.prepare_column_lists()

        self.measure_activity()

        self.check_if_ever_overpaid()

        self.check_if_ever_delayed()

        self.check_if_ever_overdraft()

        self.measure_maximum_delay()

        self.measure_longest_streak()

        self.dump_csv()

        print("Stage finished: Feature Engineering")
