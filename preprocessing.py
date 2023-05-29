import pandas as pd

from config import PATH_TO_TEMPORARY_DATA


class PreprocessingStage:

    def __init__(self):
        self.data = pd.DataFrame

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

    def dump_csv(self):
        """
        This function dumps the data for oncoming stages
        :return: This function returns nothing.
        """
        print("Dumping...")

        self.data.to_csv(PATH_TO_TEMPORARY_DATA, index=False)

        print("Dumped.")

    def group_unlabeled_values(self):

        self.data["EDUCATION"] = self.data["EDUCATION"].replace([0, 5, 6], 4)

        self.data["MARRIAGE"] = self.data["MARRIAGE"].replace([0], 3)

    def normalize_columns(self):

        columns_to_normalize = ["LIMIT_BAL", "AGE", "SEX", "MAX_DELAY", "LONGEST_STREAK",
                                "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

        for current_column in columns_to_normalize:
            print(f"Normalizing {current_column}...")

            min_value = self.data[current_column].min()
            max_value = self.data[current_column].max()
            self.data[current_column] = (self.data[current_column] - min_value) / (max_value - min_value)

    def one_hot_encode_categorical_variables(self):
        print("One hot encoding categorical variables...")

        categorical_features = ["SEX", "EDUCATION", "MARRIAGE", "OVERPAID", "DELAYED", "OVERDRAFT"]

        for current_feature in categorical_features:
            print(f"Encoding {current_feature}...")

            current_dummy = pd.get_dummies(self.data[current_feature], prefix=current_feature)

            self.data = pd.concat([self.data, current_dummy], axis=1)

        self.data.drop(columns=categorical_features, inplace=True)

        print("One hot encoding complete.")

    def preprocess(self):

        self.group_unlabeled_values()

        self.normalize_columns()

        self.one_hot_encode_categorical_variables()

    def execute_stage(self):
        """
        This function executes the current stage.
        :return: This function returns nothing.
        """

        print("Beginning stage: Preprocessing")

        self.load_data()

        self.preprocess()

        self.dump_csv()

        print("Stage finished: Preprocessing")
