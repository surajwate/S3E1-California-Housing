import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans


def kfold(file_path, n_splits=5, shuffle=True, random_state=42, save_path=None):
    """
    Creates K-Fold indices for a dataset loaded from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    n_splits (int): Number of folds. Default is 5.
    shuffle (bool): Whether to shuffle the data. Default is True.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. If None, the file is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, "kfold"] = fold

    if save_path:
        data.to_csv(save_path, index=False)

    return data


def classification_stratified(
    file_path, target_column, n_splits=5, random_state=42, save_path=None
):
    """
    Creates stratified K-Fold indices for classification tasks from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    target_column (str): The name of the target column.
    n_splits (int): Number of folds. Default is 5.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. If None, the file is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1
    y = data[target_column].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    if save_path:
        data.to_csv(save_path, index=False)

    return data


def regression_stratified(
    file_path,
    target_column,
    n_splits=5,
    binning_method="sturges",
    custom_bins=None,
    random_state=42,
    save_path=None,
):
    """
    Creates stratified K-Fold indices for regression tasks with various binning methods from a CSV file.

    Parameters:
    file_path (str): Path to the input CSV file containing the dataset.
    target_column (str): The name of the target column.
    n_splits (int): Number of folds. Default is 5.
    binning_method (str): Method for binning the target variable. Options are 'sturges', 'quantile', 'kmeans', 'custom'. Default is 'sturges'.
    custom_bins (list): List of bin edges for custom binning. Required if binning_method is 'custom'.
    random_state (int): Seed for the random number generator. Default is 42.
    save_path (str): Optional path to save the CSV file. If None, the file is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'kfold' column.
    """
    data = pd.read_csv(file_path)
    data["kfold"] = -1

    if binning_method == "sturges":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.cut(data[target_column], bins=num_bins, labels=False)
    elif binning_method == "quantile":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data["bins"] = pd.qcut(data[target_column], q=num_bins, labels=False)
    elif binning_method == "kmeans":
        num_bins = int(np.floor(1 + np.log2(len(data))))
        kmeans = KMeans(n_clusters=num_bins, random_state=random_state)
        data["bins"] = kmeans.fit_predict(data[[target_column]])
    elif binning_method == "custom":
        if custom_bins is None:
            raise ValueError("Custom bins must be provided when using custom binning.")
        data["bins"] = pd.cut(
            data[target_column], bins=custom_bins, labels=False, include_lowest=True
        )
    else:
        raise ValueError(
            f"Invalid binning method: {binning_method}. Choose 'sturges', 'quantile', 'kmeans', or 'custom'."
        )

    y = data["bins"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y)):
        data.loc[val_idx, "kfold"] = fold

    data = data.drop("bins", axis=1)

    if save_path:
        data.to_csv(save_path, index=False)

    return data


"""
Example Usage:

from create_folds import kfold, classification_stratified, regression_stratified

# Example usage for K-Fold
kfold_file = kfold(file_path="train.csv", n_splits=5, shuffle=True, random_state=42)
print(f"K-Fold data saved to: {kfold_file}")

# Example usage for Stratified K-Fold Classification
stratified_classification_file = classification_stratified(file_path="train.csv", target_column="target", n_splits=5, random_state=42)
print(f"Stratified Classification data saved to: {stratified_classification_file}")

# Example usage for Stratified K-Fold Regression with Sturges' binning
stratified_regression_sturges_file = regression_stratified(file_path="train.csv", target_column="target", n_splits=5, binning_method='sturges', random_state=42)
print(f"Stratified Regression with Sturges' binning data saved to: {stratified_regression_sturges_file}")

# Example usage for Stratified K-Fold Regression with custom binning
custom_bins = [0, 1, 2, 3, 4.841, 5.1]
stratified_regression_custom_file = regression_stratified(file_path="train.csv", target_column="target", n_splits=5, binning_method='custom', custom_bins=custom_bins, random_state=42)
print(f"Stratified Regression with custom binning data saved to: {stratified_regression_custom_file}")

"""
