import pickle
import pandas as pd
from source.logger import logging
from source.exception import ChurnException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from source.utility.utility import import_csv_file


def tune_hyperparameters(x_train, y_train):
    try:
        model = GradientBoostingClassifier()

        # Define hyperparameters to tune
        # param_grid = {
        #     'loss': ['deviance', 'exponential'],
        #     'learning_rate': [0.01, 0.1, 0.5],
        #     'n_estimators': [50, 100, 200],
        #     'subsample': [0.5, 1.0],
        #     'criterion': ['friedman_mse', 'mse'],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2],
        #     'min_weight_fraction_leaf': [0.0, 0.1],
        #     'max_depth': [3, 5, 7],
        #     'min_impurity_decrease': [0.0, 0.1],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'warm_start': [False, True],
        #     'validation_fraction': [0.1, 0.2]
        # }

        param_grid = {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.01], # [0.01, 0.1, 0.5]
            'n_estimators': [5, 10] # [50, 100]
        }

        # Create F1 scorer
        f1_scorer = make_scorer(f1_score, average='macro')

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=f1_scorer)

        # Fit the GridSearchCV object to the training data
        grid_search.fit(x_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score

    except ChurnException as e:
        raise e


class ModelTrain:
    def __init__(self, utility_config):
        try:

            self.utility_config = utility_config
            # Dictionary containing models
            self.models = {
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
            self.model_evaluation_report = pd.DataFrame(columns=['model_name', 'accuracy', 'recall', 'precision', 'f1_score', 'conf_matrix'])

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
            self.model_evaluation_report = self.model_evaluation_report._append(pd.Series(new_row, index=self.model_evaluation_report.columns), ignore_index=True)

        except ChurnException as e:
            raise e

    def model_training(self, train_data, test_data):
        try:
            x_train = train_data.drop('Churn', axis=1)
            y_train = train_data['Churn']
            x_test = test_data.drop('Churn', axis=1)
            y_test = test_data['Churn']

            # Iterate through each model
            for name, model in self.models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                with open(f"{self.utility_config.mt_model_path}/{name}.pkl", 'wb') as f:
                    pickle.dump(model, f)

                self.log_and_update_metrics(y_test, y_pred, name)

        except ChurnException as e:
            raise e

    def finalize_model(self):
        try:
            self.model_evaluation_report.sort_values(by='f1_score', ascending=False, inplace=True)

            # Log column names and corresponding values from the first row
            first_row = self.model_evaluation_report.iloc[0]
            log_message = "Model Evaluation Report:\t"
            for column_name, value in first_row.items():
                log_message += f"{column_name}: {value}\t"

            logging.info(log_message)

            return first_row['model_name']

        except ChurnException as e:
            raise e

    def retrain_final_model(self, final_model_name, train_data, test_data):
        try:
            x_train = train_data.drop(self.utility_config.target_column, axis=1)
            y_train = train_data[self.utility_config.target_column]

            # Obtain the best parameters and score using tune_hyperparameters function
            best_params, _ = tune_hyperparameters(x_train, y_train)

            # Instantiate the final model with the best parameters
            final_model = GradientBoostingClassifier(**best_params)

            # Train the final model on the entire training dataset
            final_model.fit(x_train, y_train)

            # Optionally, evaluate the final model on test data
            x_test = test_data.drop(self.utility_config.target_column, axis=1)
            y_test = test_data[self.utility_config.target_column]
            test_score = final_model.score(x_test, y_test)
            logging.info(f"Final model ({final_model_name}) test score: {test_score}")

            with open(f"{self.utility_config.mt_final_model}/{final_model_name}.pkl", 'wb') as f:
                pickle.dump(final_model, f)

        except ChurnException as e:
            raise e

    def initiate_model_training(self):
        try:
            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.dt_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.dt_test_file_path)

            self.model_training(train_data, test_data)
            self.model_evaluation_report.to_csv(self.utility_config.mt_model_path + '/model_evaluation_report.csv', index=False)

            final_model_name = self.finalize_model()

            self.retrain_final_model(final_model_name, train_data, test_data)

            print('done')
        except ChurnException as e:
            raise e

