# Build a ML Model using bagging technique

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from create_fold import regression_stratified
from sklearn.ensemble import BaggingRegressor


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
    # scalar = StandardScaler()
    # X_train = scalar.fit_transform(X_train)
    # X_test = scalar.transform(X_test)

    # Initialize a Decision Tree Classifier
    base_estimator = DecisionTreeRegressor(random_state=42)

    # Initialize BaggingClassifier with the Decision Tree as the base estimator
    bagging_clf = BaggingRegressor(estimator=base_estimator, 
                                    n_estimators=50, 
                                    random_state=42)

    # Train the BaggingClassifier
    bagging_clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = bagging_clf.predict(X_test)

    # Check the performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Fold={fold}, RMSE={rmse}")

    return rmse


if __name__ == "__main__":
    methods = ['sturges', 'quantile', 'kmeans']
    for method in methods:
        df = regression_stratified(file_path='./input/train.csv', target_column='MedHouseVal', n_splits=5, binning_method=method)
        print(f"Method: {method}")
        average_rmse = 0
        for fold_ in range(5):
            rmse = run(df, fold_)
            average_rmse += rmse
        print(f"Average RMSE using {method} for binning the MedHouseVal: {average_rmse/5}")
