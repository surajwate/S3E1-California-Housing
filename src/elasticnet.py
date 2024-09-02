# Linear regression model with stratified k-fold cross-validation
# Log Transform the target variable

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from create_fold import regression_stratified
from sklearn.model_selection import GridSearchCV


def run(df, fold):
    # Load the training data with folds
    # df = pd.read_csv("./input/train_folds.csv")

    # Split the data into training and validation sets
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # Split the feature and target
    X_train = df_train.drop(columns=['id', 'kfold', 'MedHouseVal'], axis=1)
    y_train = df_train['MedHouseVal']
    X_test = df_test.drop(columns=['id', 'kfold', 'MedHouseVal'], axis=1)
    y_test = df_test['MedHouseVal']

    # Apply log transformation to the target variable
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # All features are numerical, therefore we will scale all features
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [10000],  # Increase the number of iterations significantly
        'tol': [0.001]  # Optional: Adjust tolerance to allow earlier stopping
    }

    # Initialize the model
    model = ElasticNet()

    # Set up grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # Fit the model
    grid_search.fit(X_train, y_train_log)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Use the best model found by grid search
    best_model = grid_search.best_estimator_

    # Prediction on log scale
    y_pred_log = best_model.predict(X_test)

    # Inverse transform the predictions back to original scale
    y_pred = np.expm1(y_pred_log)   # expm1 is the inverse of log1p

    # Check the performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Fold={fold}, RMSE={rmse}")
    return rmse


if __name__ == "__main__":
    # methods = ['sturges', 'quantile', 'kmeans']
    # for method in methods:
    method = 'kmeans'
    df = regression_stratified(file_path='./input/train.csv', target_column='MedHouseVal', n_splits=5, binning_method=method)
    average_rmse = 0
    print(f"Method: {method}")
    for fold_ in range(5):
        rmse = run(df, fold_)
        average_rmse += rmse
    # print(f"RMSE using {method} for binning the MedHouseVal: {run(df, 0)}")

    print(f"Average RMSE using {method} for binning the MedHouseVal: {average_rmse/5}")