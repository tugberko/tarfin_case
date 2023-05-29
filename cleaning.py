from typing import Union

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import PATH_TO_TEMPORARY_DATA, RANDOM_SEED


class DataCleaningStage:

    def __init__(self, outlier_removal_method: int = 0):

        self.data: Union[None, pd.DataFrame] = None

        if outlier_removal_method == 0:
            self.outlier_removal_method = "lof"
        else:
            self.outlier_removal_method = "isolation_forest"

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

    def measure_imbalance(self):
        """
        This function displays the fractions of majority and minority classes.
        :return: This function returns nothing
        """

        defaulter_frac = self.data["DEFAULT"].mean()

        print(f"Imbalance: {defaulter_frac} / {1 - defaulter_frac}")

    def remove_outliers(self):
        """
        This function removes local outliers from data.
        :return: This function returns nothing.
        """

        print("Removing outliers...")

        outlier_detector = None

        # Default
        if self.outlier_removal_method == "lof":
            print("Method: Local Outlier Factor")
            outlier_detector = LocalOutlierFactor()

        if self.outlier_removal_method == "isolation_forest":
            print("Method: Isolation forest")
            if RANDOM_SEED:
                outlier_detector = IsolationForest(random_state=RANDOM_SEED)
            else:
                outlier_detector = IsolationForest


        outlier_scores = outlier_detector.fit_predict(self.data)

        outlier_indices = self.data.index[outlier_scores == -1]

        num_outliers = len(outlier_indices)

        shrinkage = num_outliers / len(self.data) * 100

        print(f"{num_outliers} outliers has been detected. ({shrinkage}% shrinkage)")

        self.data.drop(outlier_indices, inplace=True)

        print("Removed outliers.")

    def impute_missing_values(self):
        """
        This function is used for imputing missing values (If we had any)
        :return: This function returns nothing
        """
        pass

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

        print("Beginning stage: Data Cleaning")

        self.load_data()

        self.measure_imbalance()

        self.remove_outliers()

        self.measure_imbalance()

        self.impute_missing_values()

        self.dump_csv()

        print("Stage finished: Data Cleaning")
