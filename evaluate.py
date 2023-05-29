from typing import Union, List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import PATH_TO_TEMPORARY_DATA, TRAIN_TEST_SPLIT_RATIO, BETA


class EvaluationStage:

    def __init__(self):

        self.input_features: Union[None, List] = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

        self.data: Union[None, pd.DataFrame] = None

        self.target_variable: str = "DEFAULT"

        self.models: List = []

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

    def prepare_input_features(self):
        """
        This function initializes the input features.
        :return: This function returns nothing.
        """
        print("Preparing input features...")

        self.input_features: List = self.data.columns.tolist()

        print(f"Columns found: {self.input_features}")

        print("Removing target variable...")
        self.input_features.remove(self.target_variable)
        print("Target variable removed.")

        print("Input features ready.")

    def split_data(self):
        """
        This function splits the train and test data.
        :return: This function returns nothing.
        """

        print("Splitting data...")

        X = self.data[self.input_features]
        y = self.data[self.target_variable]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO)

        print("Data split.")

    def initalize_models(self):
        """
        This function initializes the models we want to evaluate
        :return: This function returns nothing.
        """

        print("Initializing models...")

        xgb_clf = xgboost.XGBClassifier()
        xgb = {
            "name": "XGBoost",
            "classifier": xgb_clf,
            "param_grid": {
                "eta": [1]
            }
        }
        # self.models.append(xgb)

        dt_clf = DecisionTreeClassifier()
        decision_tree = {
            "name": "Decision Tree",
            "classifier": dt_clf,
            "param_grid": {
                "criterion": ["entropy"]
            }
        }
        self.models.append(decision_tree)

        rf_clf = RandomForestClassifier()
        random_forest = {
            "name": "Random Forest",
            "classifier": rf_clf,
            "param_grid": {
                "criterion": ["entropy"]
            }
        }
        self.models.append(random_forest)

        adaboost_clf = AdaBoostClassifier()
        adaboost = {
            "name": "AdaBoost",
            "classifier": adaboost_clf,
            "param_grid": {
                "learning_rate": [1.0]
            }
        }
        self.models.append(adaboost)

        knn_clf = KNeighborsClassifier()
        knn = {
            "name": "KNN",
            "classifier": knn_clf,
            "param_grid": {
                "n_neighbors": [5]
            }
        }
        self.models.append(knn)

        qda_clf = QuadraticDiscriminantAnalysis()
        qda = {
            "name": "Quadratic Discriminant Analysis",
            "classifier": qda_clf,
            "param_grid": {
                "reg_param": [0]
            }
        }
        self.models.append(qda)

        gnb_clf = GaussianNB()
        gnb = {
            "name": "Gaussian Naive Bayes",
            "classifier": gnb_clf,
            "param_grid": {
                "var_smoothing": [1e-9]
            }
        }
        self.models.append(gnb)

        print(f"{len(self.models)} models has been initialized.")

    @staticmethod
    def fbeta_scoring(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=BETA)  # Adjust the beta value as needed

    def select_features(self, estimator, n):
        print("Selecting features...")
        rfe = RFE(estimator, n_features_to_select=n, verbose=True)

        rfe.fit(self.X_train, self.y_train)

        selected_features = self.X_train.columns[rfe.support_]
        print(f"Selected features: {selected_features}")
        return selected_features

    def evaluate_model(self, model: Dict):
        """
        This stage evaluates a model.
        :param model:
        :return:
        """

        # Model information
        classifier_name = model["name"]
        classifier = model["classifier"]
        parameter_grid = model["param_grid"]

        print(f"\nClassifier: {classifier_name}")

        # Defining cross validation strategy
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)

        # Defining custom f-beta scorer
        scorer = make_scorer(self.fbeta_scoring)

        # Hyperparameter tuning with cross validated grid search
        grid_search = GridSearchCV(estimator=classifier, param_grid=parameter_grid, cv=cv, scoring=scorer)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best hyperparameter values and model performance
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        # Print some output
        print(f"[Validation] Best F{BETA}:\t{best_score}")
        print(f"Best parameters:")
        print(best_params)

        # Feature selection
        try:
            best_features = self.select_features(best_model, 25)
        except ValueError:
            print("Feature selection is not available for this classifier.")
            best_features = self.input_features

        # Test section
        best_model.fit(self.X_train[best_features], self.y_train)

        predictions = best_model.predict(self.X_test[best_features])

        # Results

        # Save the ROC to the disk
        RocCurveDisplay.from_estimator(best_model, self.X_test[best_features], self.y_test, name=classifier_name)
        plt.savefig(f"ROC-{str(type(best_model))}.png")
        plt.title(str(type(best_model)))
        plt.clf()

        test_score = self.fbeta_scoring(self.y_test, predictions)
        print(f"[Test] F{BETA}:\t{test_score}")

        test_accuracy = accuracy_score(self.y_test, predictions)
        print(f"[Test] Accuracy:\t{test_accuracy}")

        cm = confusion_matrix(self.y_test, predictions)
        print(f"Confusion matrix:\n{cm}\n\n")

        # Save confusion matrix to disk
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot()

        plt.title(str(type(best_model)))
        plt.savefig(f"CM-{str(type(best_model))}.png")
        plt.clf()

    def execute_stage(self):

        print("Beginning stage: Model Evaluation")

        self.load_data()

        self.prepare_input_features()

        self.split_data()

        self.initalize_models()

        for current_model in self.models:
            self.evaluate_model(current_model)

        print("Stage finished: Model Evaluation")
