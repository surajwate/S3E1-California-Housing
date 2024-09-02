# Build a ML Model using Gradient Boosting and RandomizedSearchCV technique

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from create_fold import regression_stratified
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import time
from xgboost import XGBRegressor


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

    # All features are numerical, therefore we will scale all features
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Apply log transformation to the target variable
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)


    # Define the parameter distributions for RandomizedSearch
    param_dist = {
    'n_estimators': randint(290, 320),  # Slightly narrow around 303
    'learning_rate': uniform(0.03, 0.01),  # Narrow around 0.038
    'max_depth': randint(7, 9),  # Narrow around 8
    'min_child_weight': randint(3, 5),  # Narrow around 4
    'subsample': uniform(0.9, 0.05),  # Narrow around 0.922
    'colsample_bytree': uniform(0.7, 0.05),  # Narrow around 0.736
    'gamma': uniform(0.04, 0.01),  # Narrow around 0.045
    'reg_alpha': uniform(0.001, 0.001),  # Narrow around 0.0017
    'reg_lambda': uniform(1.5, 0.2)  # Narrow around 1.69
    }

    # Initialize the Gradient Boosting Regressor
    model = XGBRegressor(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=3
    )

    # Fit the random search
    random_search.fit(X_train, y_train_log)

    # Get the best parameters
    best_params = random_search.best_params_

    # get the best estimator
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred_log = best_model.predict(X_test)

    # Inverse transform the predictions back to original scale
    y_pred = np.expm1(y_pred_log)   # expm1 is the inverse of log1p

    # Calculate RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Fold={fold} - RMSE: {rmse:.4f}, Best Parameters: {best_params}")

    return rmse, best_params


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} minutes and {seconds} seconds"

if __name__ == "__main__":
    method = 'sturges'
    df = regression_stratified(file_path='./input/train.csv', target_column='MedHouseVal', n_splits=5, binning_method=method)
    average_rmse = 0
    results_dict = {}
    start_time = time.time()
    for fold_ in range(5):
        fold_start_time = time.time()
        rmse, best_params = run(df, fold_)
        average_rmse += rmse
        fold_time = time.time() - fold_start_time
        results_dict[fold_] = {'rmse': rmse, 'best_params': best_params}
        print(f"Time taken for fold {fold_}: {format_time(fold_time)}")
    
    total_time = time.time() - start_time
    print(f"\nAverage RMSE using {method} for binning the MedHouseVal: {average_rmse/5:.4f}")
    print(f"Total Time taken: {format_time(total_time)}")
    
    print("\nResults for each fold:")
    for fold, result in results_dict.items():
        print(f"  Fold {fold}: RMSE={result['rmse']:.4f}, Best Parameters={result['best_params']}")


