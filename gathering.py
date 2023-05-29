from typing import Union

import pandas as pd

from config import PATH_TO_RAW_DATA, PATH_TO_TEMPORARY_DATA


class DataGatheringStage:

    def __init__(self):

        self.data: Union[None, pd.DataFrame] = None

    def read_raw_data(self):
        """
        This function reads raw data from disk.
        :return: This function returns nothing.
        """

        try:
            print("Loading raw data...")
            self.data = pd.read_excel(PATH_TO_RAW_DATA, header=[1])
            print("Raw data has been loaded.")
        except FileNotFoundError:
            print(f"No such file {PATH_TO_RAW_DATA}")
            quit()

    def refine_columns(self):
        """
        This function fixes few bad column names and drops irrelevant columns such as ID
        :return: This function returns nothing
        """
        print("Refining columns...")

        self.data.rename(columns={"default payment next month": "DEFAULT", "PAY_0": "PAY_1"},
                         inplace=True)

        irrelevant_columns = ["ID"]

        self.data.drop(columns=irrelevant_columns, inplace=True)

        print("Refined columns.")

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

        print("Beginning stage: Data Gathering")

        self.read_raw_data()
        self.refine_columns()
        self.dump_csv()

        print("Stage finished: Data Gathering")
