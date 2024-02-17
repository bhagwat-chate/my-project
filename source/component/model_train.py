import pickle
import pandas as pd
from source.logger import logging
from source.exception import ChurnException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


class ModelTrain:
    def __init__(self, utility_config):
        try:
            self.utility_config = utility_config
            self.model_evaluation_report = pd.DataFrame(
                columns=['model_name', 'accuracy', 'recall', 'precision', 'f1_score', 'conf_matrix'])
        except ChurnException as e:
            raise e

    def log_and_update_metrics(self, y_true, y_pred, model_name):
        try:
            accuracy = round(accuracy_score(y_true, y_pred), 2)
            precision = round(precision_score(y_true, y_pred, average='weighted'), 2)
            recall = round(recall_score(y_true, y_pred, average='weighted'), 2)
            f1 = round(f1_score(y_true, y_pred, average='weighted'), 2)
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)
            logging.info(f"model:{model_name}, accuracy:{accuracy}, precision:{precision}, "
                         f"recall:{recall}, f1:{f1}, conf_matrix:{conf_matrix}")

            new_row = [model_name, accuracy, recall, precision, f1, conf_matrix]
            self.model_evaluation_report = self.model_evaluation_report._append(
                pd.Series(new_row, index=self.model_evaluation_report.columns), ignore_index=True)

        except ChurnException as e:
            raise e

    def model_training(self, train_data, test_data):
        try:
            x_train = train_data.drop('Churn', axis=1)
            y_train = train_data['Churn']
            x_test = test_data.drop('Churn', axis=1)
            y_test = test_data['Churn']

            # Dictionary containing models
            models = {
                "Logistic_Regression": LogisticRegression(),
                "SVM": SVC(),
                "Decision_Trees": DecisionTreeClassifier(),
                "Random_Forest": RandomForestClassifier(),
                "Gradient_Boosting_Machines": GradientBoostingClassifier(),
                "Naive_Bayes": GaussianNB(),
                "K_Nearest_Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            # Iterate through each model
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                with open(f"{self.utility_config.model_path}/{name}.pkl", 'wb') as f:
                    pickle.dump(model, f)

                self.log_and_update_metrics(y_test, y_pred, name)

        except ChurnException as e:
            raise e

    def initiate_model_training(self):
        try:
            train_data = pd.read_csv(self.utility_config.dt_train_file_path, dtype={'TotalCharges': 'float64'})
            test_data = pd.read_csv(self.utility_config.dt_test_file_path, dtype={'TotalCharges': 'float64'})

            self.model_training(train_data, test_data)
            self.model_evaluation_report.to_csv(self.utility_config.model_path + '/model_evaluation_report.csv', index=False)

        except ChurnException as e:
            raise e
