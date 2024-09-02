# Analysis of the results

## Setting: 1

**File**: `base_reg.py`

**Preprocessing:**:

- Standard Scaling on all features.

**Model**:

- Linear Regression

**Results**:

```bash
Fold=0, RMSE=0.7279232966617135
Fold=1, RMSE=1.660613388489584
Fold=2, RMSE=0.7365203142759256
Fold=3, RMSE=0.7217601446619399
Fold=4, RMSE=0.7326946244306745
```

RMSE of fold 1 is very high compared to other folds. I decduced that the target variable have outliers, which is causing creation of one fold with high RMSE.
I tried different methods of creating folds, by using different ways of binning the target variable, the results are as below:

```bash
Method: sturges
Fold=0, RMSE=0.7279232966617135
Fold=1, RMSE=1.660613388489584
Fold=2, RMSE=0.7365203142759256
Fold=3, RMSE=0.7217601446619399
Fold=4, RMSE=0.7326946244306745
Average RMSE using sturges for binning the MedHouseVal: 0.9159023537039677
Method: quantile
Fold=0, RMSE=0.7405991448514753
Fold=1, RMSE=0.7159512425706072
Fold=2, RMSE=0.7322426866235475
Fold=3, RMSE=0.726533387729987
Fold=4, RMSE=1.5852165667461742
Average RMSE using quantile for binning the MedHouseVal: 0.9001086057043584
Method: kmeans
Fold=0, RMSE=0.7310631887078294
Fold=1, RMSE=0.7195423381855298
Fold=2, RMSE=0.7305703526775712
Fold=3, RMSE=0.7175825132280376
Fold=4, RMSE=1.6316117960028385
Average RMSE using kmeans for binning the MedHouseVal: 0.9060740377603613
```

In all the methods, the RMSE of one fold is very high compared to other folds.

### Possible Solutions

1. Transform the Target Variable
    - Log Transformation
    - Box-Cox Transformation
2. Robust Regression Models
    - Ridge Regression
    - Lasso Regression
    - Elastic Net Regression
    - Huber Regression
3. Remove or Cap Outliers

    ```python
    df['MedHouseVal'] = np.where(df['MedHouseVal'] > 4.8, 4.8, df['MedHouseVal'])
    ```

4. Stratified Cross-Validation with Consideration for Outliers

## Setting 2: After Applying Log Transformation

**File**: `logtransform_reg.py`

**Preprocessing**:

- Applied Log Transformation on Target Variable
- Standard Scaling on all features

**Model**:

- Linear Regression

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.8084585111061706
Fold=1, RMSE=0.7904659986993912
Fold=2, RMSE=0.8404356136641057
Fold=3, RMSE=0.8048269679099952
Fold=4, RMSE=0.8267859088894125
Average RMSE using sturges for binning the MedHouseVal: 0.8141946000538149
Method: quantile
Fold=0, RMSE=0.824744571360597
Fold=1, RMSE=0.8027671844783031
Fold=2, RMSE=0.8261737521824027
Fold=3, RMSE=0.8576201618283532
Fold=4, RMSE=0.7665117480400239
Average RMSE using quantile for binning the MedHouseVal: 0.815563483577936
Method: kmeans
Fold=0, RMSE=0.8162667398649385
Fold=1, RMSE=0.8086714194618515
Fold=2, RMSE=0.8246023377134531
Fold=3, RMSE=0.8000973192317443
Fold=4, RMSE=0.8184038333065945
Average RMSE using kmeans for binning the MedHouseVal: 0.8136083299157164
```

After applying log transformation, the RMSE of all folds are much more consistent across all folds. This indicates that the log transformation has helped to stabilize the model's performance, particularly in handling the skewness and outliers in the target variable.

## Setting 3: Regularization

**File**: `l1l2.py`

**Preprocessing**:

- Applied Log Transformation on Target Variable
- Standard Scaling on all features

### Ridge Regression

alpha=1.0

**Results**:

```bash
Fold=0, RMSE=0.8162957463254276
Fold=1, RMSE=0.8086916583893567
Fold=2, RMSE=0.8246305680055753
Fold=3, RMSE=0.8001230660300255
Fold=4, RMSE=0.8184316021930738
Average RMSE using kmeans for binning the MedHouseVal: 0.8136345281886918
```

### Lasso Regression

alpha=0.1

**Results**:

```bash
Fold=0, RMSE=0.9184411313438348
Fold=1, RMSE=0.9062454067316098
Fold=2, RMSE=0.9135182499747173
Fold=3, RMSE=0.9064692591388754
Fold=4, RMSE=0.9209086297283678
Average RMSE using kmeans for binning the MedHouseVal: 0.9131165353834809
```

Between Ridge and Lasso, Ridge Regression has a lower RMSE compared to Lasso Regression. This is expected as Lasso Regression tends to perform feature selection by setting some coefficients to zero, which may not be ideal in this case.

## Setting 4: Elastic Net Regression with Grid Search

**File**: `elasticnet.py`

**Preprocessing**:

- Applied Log Transformation on Target Variable
- Standard Scaling on all features

**Results**:

```bash
Best parameters: {'alpha': 0.1, 'l1_ratio': 0.1, 'max_iter': 10000, 'tol': 0.001}
Fold=0, RMSE=0.8491065036139921
Best parameters: {'alpha': 0.1, 'l1_ratio': 0.1, 'max_iter': 10000, 'tol': 0.001}
Fold=1, RMSE=0.8281935333466821
Best parameters: {'alpha': 0.01, 'l1_ratio': 0.9, 'max_iter': 10000, 'tol': 0.001}
Fold=2, RMSE=0.836695611463773
Best parameters: {'alpha': 0.1, 'l1_ratio': 0.1, 'max_iter': 10000, 'tol': 0.001}
Fold=3, RMSE=0.8283874129359954
Best parameters: {'alpha': 0.0001, 'l1_ratio': 0.1, 'max_iter': 10000, 'tol': 0.001}
Fold=4, RMSE=0.8184764336253199
Average RMSE using kmeans for binning the MedHouseVal: 0.8321718989971526
```

Elastic Net Regression has a lowest average RMSE compared to Ridge and Lasso Regression. This is expected as Elastic Net combines the penalties of Lasso and Ridge Regression, which allows it to perform better in handling multicollinearity and feature selection.
