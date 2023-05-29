from typing import Union

import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import resample

from config import PATH_TO_TEMPORARY_DATA, RANDOM_SEED


class DataBalancingStage:

    def __init__(self, method: str = "smote"):

        self.data: Union[None, pd.DataFrame] = None

        self.available_methods = ["random_undersample", "smote", "adasyn"]

        self.method = self.available_methods[0]
        if method.lower() in self.available_methods:
            self.method = method.lower()

    def load_data(self):
        """
        This function is used for reading the data produced by previous stage.
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

    def random_undersample_majorities(self):
        """
        This function randomlv undersamples the majority class.
        :return: This function returns nothing
        """

        print("Balancing data using random undersampling of majority class...")

        majority_class = self.data[self.data["DEFAULT"] == self.data["DEFAULT"].value_counts().idxmax()]
        minority_class = self.data[self.data["DEFAULT"] != self.data["DEFAULT"].value_counts().idxmax()]

        if RANDOM_SEED:
            undersampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class),
                                             random_state=RANDOM_SEED)
        else:
            undersampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class))

        self.data = pd.concat([undersampled_majority, minority_class])

        print("Balancing complete.")

    def oversample_minorities_using_smote(self):
        """
        This function oversamples the minorities using SMOTE.
        :return: This function returns nothing.
        """

        print("Balancing data using SMOTE...")

        if RANDOM_SEED:
            smote = SMOTE(random_state=RANDOM_SEED)
        else:
            smote = SMOTE()

        target_variable = "DEFAULT"
        input_features = self.data.columns.tolist()
        input_features.remove(target_variable)

        X = self.data[input_features]
        y = self.data[target_variable]

        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.data = pd.concat([X_resampled, y_resampled], axis=1)

        print("Balancing complete.")

    def oversample_minorities_using_adasyn(self):
        """
        This function oversamples the minorities using ADASYN.
        :return: This function returns nothing.
        """

        print("Balancing data using ADASYN...")

        if RANDOM_SEED:
            adasyn = ADASYN(random_state=RANDOM_SEED)
        else:
            adasyn = ADASYN()

        target_variable = "DEFAULT"
        input_features = self.data.columns.tolist()
        input_features.remove(target_variable)

        X = self.data[input_features]
        y = self.data[target_variable]

        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        self.data = pd.concat([X_resampled, y_resampled], axis=1)

        print("Balancing complete.")

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

        print("Beginning stage: Data Balancing")

        self.load_data()

        self.measure_imbalance()

        if self.method == "random_undersample":
            self.random_undersample_majorities()

        if self.method == "smote":
            self.oversample_minorities_using_smote()

        if self.method == "adasyn":
            self.oversample_minorities_using_adasyn()

        self.dump_csv()

        print("Stage finished: Data Balancing")
