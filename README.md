# California Housing Price Prediction

This project is aimed at predicting housing prices in California using various machine learning models. The dataset used for this project is derived from the famous California Housing dataset, which contains information about different housing attributes and their corresponding median values.

## Project Structure

The project is organized into the following directories and files:

```plaintext
S3E1 CALIFORNIA HOUSING/
│
├── docs/
│   └── Notes.md                      # Markdown file for keeping track of notes and observations
│
├── env/                              # Python virtual environment directory (not included in the repo)
│
├── input/
│   ├── sample_submission.csv         # Sample submission file for Kaggle competition
│   ├── test.csv                      # Test dataset for making predictions
│   ├── train_folds.csv               # Training dataset with k-fold splits
│   └── train.csv                     # Original training dataset
│
├── notebooks/
│   ├── eda.ipynb                     # Jupyter notebook for Exploratory Data Analysis (EDA)
│   ├── explore_data.ipynb            # Notebook for exploring data and feature engineering
│   └── linear_regression.ipynb       # Notebook for building and testing Linear Regression model
│
├── src/                              # Source code for model training and evaluation
│   ├── analyze.py                    # Script for analyzing model performance
│   ├── bagging.py                    # Script for training a Bagging Regressor model
│   ├── base_reg.py                   # Script for training a baseline regression model
│   ├── create_fold.py                # Script for creating stratified k-folds
│   ├── dtr.py                        # Script for training a Decision Tree Regressor model
│   ├── elasticnet.py                 # Script for training an Elastic Net Regressor model
│   ├── gradient_gridsearch.py        # Script for training Gradient Boosting with GridSearchCV
│   ├── gradient_randomizedsearch.py  # Script for training Gradient Boosting with RandomizedSearchCV
│   ├── gradient.py                   # Script for training a basic Gradient Boosting Regressor
│   ├── l1l2.py                       # Script for training Ridge and Lasso models
│   ├── logtransform_reg.py           # Script for regression after log-transforming the target
│   ├── temp.py                       # Temporary script for testing various models and approaches
│   ├── xgb_random.py                 # Script for training XGBoost with RandomizedSearchCV
│   └── xgb.py                        # Script for training a basic XGBoost model
│
└── .gitignore                        # Git ignore file to exclude unnecessary files from the repository
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Installing Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can generate one using:

```bash
pip freeze > requirements.txt
```

### Running the Project

1. **Exploratory Data Analysis (EDA)**: Start by exploring the dataset using the `notebooks/eda.ipynb` notebook. This will give you insights into the data distribution, relationships between features, and potential feature engineering steps.

2. **Preprocessing**: Use the `src/create_fold.py` script to create stratified k-folds for cross-validation. This ensures that the target variable is distributed evenly across folds.

3. **Model Training**: You can experiment with different models using the scripts in the `src/` directory. For example, to train a Gradient Boosting model with RandomizedSearchCV, run:

    ```bash
    python src/gradient_randomizedsearch.py
    ```

4. **Evaluation**: After training the models, evaluate their performance on the test set using the `src/analyze.py` script.

5. **Submission**: Once satisfied with the model's performance, use the best-performing model to generate predictions on the test set (`input/test.csv`) and submit the predictions to the Kaggle competition using the format provided in `input/sample_submission.csv`.

## Results

This section will be updated with the results of the different models and the best-performing model's configuration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for providing the dataset.
- The Scikit-learn and XGBoost teams for the powerful machine learning libraries.
- The open-source community for continuous improvements and support.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.
