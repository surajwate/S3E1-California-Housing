import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from create_fold import regression_stratified
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time

def run_grid_search(df, fold):
    # Load training data with folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # Split feature and target
    X_train = df_train.drop(columns=['id', 'kfold', 'MedHouseVal'], axis=1)
    y_train = df_train['MedHouseVal']
    X_test = df_test.drop(columns=['id', 'kfold', 'MedHouseVal'], axis=1)
    y_test = df_test['MedHouseVal']

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.052, 0.053, 0.054],
        'max_depth': [5, 6],
        'min_samples_split': [2, 3],
        'n_estimators': [250, 260, 270, 279],
        'subsample': [0.75, 0.76, 0.77]
    }

    # Initialize Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=gbr,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=3
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Fold={fold}, RMSE={rmse:.4f}")
    print(f"Best Parameters: {best_params}")

    return rmse, best_params

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} minutes and {seconds} seconds"

# Run GridSearchCV for all folds
if __name__ == "__main__":
    method = 'sturges'
    df = regression_stratified(file_path='./input/train.csv', target_column='MedHouseVal', n_splits=5, binning_method=method)
    average_rmse = 0
    results_dict = {}
    start_time = time.time()
    for fold_ in range(5):
        fold_start_time = time.time()
        rmse, best_params = run_grid_search(df, fold_)
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