"""
    date:       2019/12/29 10:39 下午
    written by: neonleexiang
"""
import DataPreprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import numpy as np


# import os
# import pandas as pd
#
#
# HOUSING_PATH = "datasets/housing"       # path of storage
#
#
# # function to change the tar file into csv file
# def load_housing_data(housing_path=HOUSING_PATH):
#     csv_path = os.path.join(housing_path, "housing.csv")
#     return pd.read_csv(csv_path)

def best_model_of_random_forest(data, labels):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(data, labels)

    return grid_search.best_estimator_


if __name__ == '__main__':
    housing = DataPreprocessing.load_housing_data()

    data_training, data_training_labels, data_test, data_test_labels = DataPreprocessing.data_preprocessing(housing)

    # training_model = best_model_of_random_forest(data_training, data_training_labels)

    # joblib.dump(training_model, "random_forest_model_for_housing_data.pkl")

    final_model = joblib.load("random_forest_model_for_housing_data.pkl")

    final_predictions = final_model.predict(data_test)

    final_mse = mean_squared_error(data_test_labels, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print(final_rmse)






