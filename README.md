# California Housing Price Prediction

This repository contains a machine learning project aimed at predicting California housing prices using the dataset provided by the [Kaggle Playground Series - Season 3, Episode 1](https://www.kaggle.com/competitions/playground-series-s3e1).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Submission](#submission)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to build a robust machine-learning model to predict median house values in California districts based on several features. The project explores various modeling techniques, including linear regression, decision trees, bagging, gradient boosting, and XGBoost, combined with hyperparameter optimization using RandomizedSearchCV and GridSearchCV.

## Dataset

The dataset used in this project is provided by the [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s3e1). It contains features like `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and the target variable `median_house_value`.

### Files

- `train.csv`: Training dataset containing features and target variables.
- `test.csv`: Test dataset to make predictions for submission.
- `sample_submission.csv`: Sample submission file format.

## Project Structure

The repository is organized as follows:

```plaintext
├── docs
│   └── Notes.md            # Notes and findings during the project
├── env                     # Virtual environment setup (if applicable)
├── input
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── train_folds.csv     # Training data split into folds (generated)
│   └── train.csv           # Original training data
├── notebooks
│   ├── eda.ipynb           # Exploratory data analysis notebook
│   ├── explore_data.ipynb  # Additional data exploration
│   └── linear_regression.ipynb  # Linear Regression trials
├── src
│   ├── analyze.py          # Scripts for data analysis
│   ├── bagging.py          # Bagging Regressor implementation
│   ├── base_reg.py         # Base model implementation (Linear Regression)
│   ├── create_fold.py      # Script to create folds for cross-validation
│   ├── dtr.py              # Decision Tree Regressor implementation
│   ├── elasticnet.py       # Elastic Net Regression implementation
│   ├── gradient.py         # Gradient Boosting implementation
│   ├── gradient_gridsearch.py  # Gradient Boosting with Grid Search
│   ├── gradient_randomizedsearch.py # Gradient Boosting with Randomized Search
│   ├── logtransform_reg.py # Linear Regression with log transformation
│   ├── temp.py             # Temporary script for testing
│   ├── xgb.py              # XGBoost Regressor implementation
│   ├── xgb_random.py       # XGBoost Regressor with Randomized Search
└── .gitignore              # Git ignore file
```

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/surajwate/S3E1-California-Housing.git
   cd S3E1-California-Housing
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Exploratory Data Analysis

Initial data exploration and visualization are performed using the notebooks located in the `notebooks/` directory. The `eda.ipynb` provides insights into the distribution of features, correlation with the target variable, and potential outliers.

## Model Training

### Cross-Validation

The dataset is split into 5 folds using stratified k-fold cross-validation. The folds are generated using the `create_fold.py` script, which ensures that the distribution of the target variable is preserved across all folds.

### Models Implemented

1. **Linear Regression**: Baseline model, implemented in `base_reg.py`.
2. **Decision Tree Regressor**: Implemented in `dtr.py`.
3. **Bagging Regressor**: Implemented in `bagging.py` with Decision Tree as the base estimator.
4. **Gradient Boosting Regressor**: Implemented in `gradient.py`.
5. **XGBoost Regressor**: Implemented in `xgb.py` and further tuned using RandomizedSearchCV in `xgb_random.py`.
6. **Elastic Net Regression**: Implemented in `elasticnet.py`.

### Hyperparameter Tuning

- **Randomized Search**: Implemented in `xgb_random.py` and `gradient_randomizedsearch.py`.
- **Grid Search**: Implemented in `gradient_gridsearch.py` (used for detailed hyperparameter tuning).

## Model Evaluation

Models are evaluated using RMSE (Root Mean Squared Error) on each fold, and the average RMSE across all folds is reported. Detailed evaluation results for each model are logged in the console and saved in the respective scripts.

## Submission

Once the best model is identified, it is trained on the full dataset, and predictions are generated for the test set. These predictions are then submitted to the Kaggle competition.

## Results

- The best model so far is the XGBoost Regressor with hyperparameter tuning using RandomizedSearchCV, achieving an average RMSE of **0.5618** across the folds.
- More detailed results are recorded in the individual scripts and the `docs/Notes.md` file.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
