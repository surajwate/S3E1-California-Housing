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

## Setting 5: Decision Tree Regressor

### Baselin DTR Model

**File**: `dtr.py`

**Preprocessing**:

- Create a data without scaling and log transformation

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.8570459852039447
Fold=1, RMSE=0.8205448792876608
Fold=2, RMSE=0.8566444803940815
Fold=3, RMSE=0.8229529674452252
Fold=4, RMSE=0.8370006421141354
Average RMSE using sturges for binning the MedHouseVal: 0.8388377908890096
Method: quantile
Fold=0, RMSE=0.8385731593748772
Fold=1, RMSE=0.8308697481397764
Fold=2, RMSE=0.8573078997822285
Fold=3, RMSE=0.8229143485119325
Fold=4, RMSE=0.8291301742434896
Average RMSE using quantile for binning the MedHouseVal: 0.8357590660104609
Method: kmeans
Fold=0, RMSE=0.8621042714139859
Fold=1, RMSE=0.8280462255793718
Fold=2, RMSE=0.8268333847759937
Fold=3, RMSE=0.83821424463132
Fold=4, RMSE=0.8426535432964761
Average RMSE using kmeans for binning the MedHouseVal: 0.8395703339394295
```

### Scaling and Log Transformation

Same settings as Baseline DTR Model, but with scaling and log transformation

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.8302853900275421
Fold=1, RMSE=0.833233908772427
Fold=2, RMSE=0.8200042834498329
Fold=3, RMSE=0.8344546728118155
Fold=4, RMSE=0.8377880077983196
Average RMSE using sturges for binning the MedHouseVal: 0.8311532525719875
Method: quantile
Fold=0, RMSE=0.8512539461304751
Fold=1, RMSE=0.8282679273829655
Fold=2, RMSE=0.8485403421603773
Fold=3, RMSE=0.8357115558204536
Fold=4, RMSE=0.834925783943486
Average RMSE using quantile for binning the MedHouseVal: 0.8397399110875515
Method: kmeans
Fold=0, RMSE=0.8561687236491462
Fold=1, RMSE=0.8272604441598833
Fold=2, RMSE=0.8380816345624439
Fold=3, RMSE=0.8244496065931866
Fold=4, RMSE=0.8502622783607605
Average RMSE using kmeans for binning the MedHouseVal: 0.8392445374650841
```

Not much improvement in RMSE after scaling and log transformation. Decision Tree Regressor is not sensitive to scaling and log transformation as it does not rely on linear relationships between features and target variable.

## Setting 6: Bagging Regressor

**File**: `bagging.py`

### Decision Tree Regressor as Base Estimator

Using Decision Tree Regressor as the base estimator

**Preprocessing**:

- No scaling and log transformation

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.5926913801969249
Fold=1, RMSE=0.5969797479955086
Fold=2, RMSE=0.6003008191967549
Fold=3, RMSE=0.5996370591269475
Fold=4, RMSE=0.605309051744048
Average RMSE using sturges for binning the MedHouseVal: 0.5989836116520368
Method: quantile
Fold=0, RMSE=0.6062935887493006
Fold=1, RMSE=0.6002524146647952
Fold=2, RMSE=0.5959842779544793
Fold=3, RMSE=0.6002741194836863
Fold=4, RMSE=0.5882870579731801
Average RMSE using quantile for binning the MedHouseVal: 0.5982182917650883
Method: kmeans
Fold=0, RMSE=0.6058726509727835
Fold=1, RMSE=0.5979967100364487
Fold=2, RMSE=0.59233433210397
Fold=3, RMSE=0.5917053729882822
Fold=4, RMSE=0.6094981469322936
Average RMSE using kmeans for binning the MedHouseVal: 0.5994814426067557
```

Bagging Regressor has a lower RMSE compared to Decision Tree Regressor. This is expected as Bagging Regressor reduces the variance of the model by training multiple models on different subsets of the data and averaging the predictions.

### Linear Regression as Base Estimator

Using Linear Regression as the base estimator

#### No Preprocessing

- No scaling and log transformation

**Results**:

```bash
└─Δ python .\src\bagging.py
Method: sturges
Fold=0, RMSE=0.715357827486027
Fold=1, RMSE=1.7098479248645904
Fold=2, RMSE=0.7265578260132294
Fold=3, RMSE=0.7092982249178451
Fold=4, RMSE=0.7231637964715747
Average RMSE using sturges for binning the MedHouseVal: 0.9168451199506533
Method: quantile
Fold=0, RMSE=0.7258021352822585
Fold=1, RMSE=0.708692549636163
Fold=2, RMSE=0.7235120191687766
Fold=3, RMSE=0.7134937568682566
Fold=4, RMSE=1.5997518388148075
Average RMSE using quantile for binning the MedHouseVal: 0.8942504599540525
Method: kmeans
Fold=0, RMSE=0.7171438261847526
Fold=1, RMSE=0.7115878630215892
Fold=2, RMSE=0.7188258167369881
Fold=3, RMSE=0.7053144206681324
Fold=4, RMSE=1.664684335547773
Average RMSE using kmeans for binning the MedHouseVal: 0.9035112524318472
```

#### With Preprocessing

- Scaling and log transformation

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.7935669755307065
Fold=1, RMSE=0.7889875668166743
Fold=2, RMSE=0.8282227060971294
Fold=3, RMSE=0.7883305988535015
Fold=4, RMSE=0.8157883693690555
Average RMSE using sturges for binning the MedHouseVal: 0.8029792433334135
Method: quantile
Fold=0, RMSE=0.8087745979863185
Fold=1, RMSE=0.7947508339096229
Fold=2, RMSE=0.8163966526314727
Fold=3, RMSE=0.8371853465370688
Fold=4, RMSE=0.7660125625744028
Average RMSE using quantile for binning the MedHouseVal: 0.804623998727777
Method: kmeans
Fold=0, RMSE=0.7988834140340219
Fold=1, RMSE=0.799131918812946
Fold=2, RMSE=0.8106143756814033
Fold=3, RMSE=0.7859912440565509
Fold=4, RMSE=0.8171508274793265
Average RMSE using kmeans for binning the MedHouseVal: 0.8023543560128499
```

## Setting 7: Gradient Boosting Regressor

**File**: `gradient.py`

**Preprocessing**:

- No preprocessing

**Results**:

```bash
└─Δ python .\src\gradient.py
Method: sturges
Fold=0, RMSE=0.5916628770773931
Fold=1, RMSE=0.5882022859195335
Fold=2, RMSE=0.5987475891730301
Fold=3, RMSE=0.5996906822527525
Fold=4, RMSE=0.6051073242488401
Average RMSE using sturges for binning the MedHouseVal: 0.5966821517343098
Method: quantile
Fold=0, RMSE=0.6025003556251141
Fold=1, RMSE=0.594051078438008
Fold=2, RMSE=0.6006799721237489
Fold=3, RMSE=0.5967855891408349
Fold=4, RMSE=0.5929013107910592
Average RMSE using quantile for binning the MedHouseVal: 0.5973836612237531
Method: kmeans
Fold=0, RMSE=0.5998526752337116
Fold=1, RMSE=0.5948259789762347
Fold=2, RMSE=0.5939655223709155
Fold=3, RMSE=0.5860913765377305
Fold=4, RMSE=0.6133913015630186
Average RMSE using kmeans for binning the MedHouseVal: 0.5976253709363222
```

### Gradient Boosting Regressor with RandomizedSearchCV

**File**: `gradient_randomizedsearch.py`

**Results**:

```bash
Fold=0 - RMSE: 0.5654, Best Parameters: {'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
Time taken for fold 0: 8 minutes and 22 seconds

Fold=1 - RMSE: 0.5613, Best Parameters: {'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
Time taken for fold 1: 8 minutes and 21 seconds

Fold=2 - RMSE: 0.5741, Best Parameters: {'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
Time taken for fold 2: 8 minutes and 25 seconds

Fold=3 - RMSE: 0.5740, Best Parameters: {'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
Time taken for fold 3: 8 minutes and 21 seconds

Fold=4 - RMSE: 0.5766, Best Parameters: {'learning_rate': np.float64(0.06494435859801283), 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 276, 'subsample': np.float64(0.7358782737814905)}
Time taken for fold 4: 8 minutes and 13 seconds

Average RMSE using sturges for binning the MedHouseVal: 0.5703
Total Time taken: 41 minutes and 44 seconds

Results for each fold:
  Fold 0: RMSE=0.5654, Best Parameters={'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
  Fold 1: RMSE=0.5613, Best Parameters={'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
  Fold 2: RMSE=0.5741, Best Parameters={'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
  Fold 3: RMSE=0.5740, Best Parameters={'learning_rate': np.float64(0.052467822135655234), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.8834959481464842)}
  Fold 4: RMSE=0.5766, Best Parameters={'learning_rate': np.float64(0.06494435859801283), 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 276, 'subsample': np.float64(0.7358782737814905)}
```
### Gradient Boosting Regressor with RandomizedSearchCV (2nd random search)

**File**: -

**Results**:

```bash
Fold=0 - RMSE: 0.5658, Best Parameters: {'learning_rate': np.float64(0.0522352538724892), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.7546708263364187)}
Time taken for fold 0: 2 minutes and 36 seconds

Fold=1 - RMSE: 0.5598, Best Parameters: {'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
Time taken for fold 1: 2 minutes and 37 seconds

Fold=2 - RMSE: 0.5719, Best Parameters: {'learning_rate': np.float64(0.07568476534011795), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 265, 'subsample': np.float64(0.7979622306417505)}
Time taken for fold 2: 2 minutes and 39 seconds

Fold=3 - RMSE: 0.5756, Best Parameters: {'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
Time taken for fold 3: 2 minutes and 46 seconds

Fold=4 - RMSE: 0.5751, Best Parameters: {'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
Time taken for fold 4: 2 minutes and 39 seconds

Average RMSE using sturges for binning the MedHouseVal: 0.5696
Total Time taken: 13 minutes and 20 seconds

Results for each fold:
  Fold 0: RMSE=0.5658, Best Parameters={'learning_rate': np.float64(0.0522352538724892), 'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 253, 'subsample': np.float64(0.7546708263364187)}
  Fold 1: RMSE=0.5598, Best Parameters={'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
  Fold 2: RMSE=0.5719, Best Parameters={'learning_rate': np.float64(0.07568476534011795), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 265, 'subsample': np.float64(0.7979622306417505)}
  Fold 3: RMSE=0.5756, Best Parameters={'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
  Fold 4: RMSE=0.5751, Best Parameters={'learning_rate': np.float64(0.05366675263676152), 'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 279, 'subsample': np.float64(0.7637017332034828)}
```

## Setting 8: XGBoost Regressor

**File**: `xgb.py`

**Results**:

```bash
Method: sturges
Fold=0, RMSE=0.580504414900342
Fold=1, RMSE=0.5711703703362706
Fold=2, RMSE=0.5927091683513392
Fold=3, RMSE=0.582210295050712
Fold=4, RMSE=0.5831421216553697
Average RMSE using sturges for binning the MedHouseVal: 0.5819472740588066
Method: quantile
Fold=0, RMSE=0.5894519402679347
Fold=1, RMSE=0.5796694038673396
Fold=2, RMSE=0.5786366778171839
Fold=3, RMSE=0.5829898082625984
Fold=4, RMSE=0.5770672407081466
Average RMSE using quantile for binning the MedHouseVal: 0.5815630141846406
Method: kmeans
Fold=0, RMSE=0.5872807335985145
Fold=1, RMSE=0.577311106128425
Fold=2, RMSE=0.5758961315095434
Fold=3, RMSE=0.5785583946763078
Fold=4, RMSE=0.5959840425284172
Average RMSE using kmeans for binning the MedHouseVal: 0.5830060816882416
```

### XGBoost Regressor with RandomizedSearchCV

**File**: `xgb_random.py`

**Results**:

```bash
Average RMSE using sturges for binning the MedHouseVal: 0.5618
Total Time taken: 2 minutes and 46 seconds

Results for each fold:
  Fold 0: RMSE=0.5585, Best Parameters={'colsample_bytree': np.float64(0.7367518666865607), 'gamma': np.float64(0.04589953290672094), 'learning_rate': np.float64(0.03824709648056803), 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 303, 'reg_alpha': np.float64(0.0017161101831750236), 'reg_lambda': np.float64(1.6923737499050842), 'subsample': np.float64(0.9227651908203118)}
  Fold 1: RMSE=0.5539, Best Parameters={'colsample_bytree': np.float64(0.7367518666865607), 'gamma': np.float64(0.04589953290672094), 'learning_rate': np.float64(0.03824709648056803), 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 303, 'reg_alpha': np.float64(0.0017161101831750236), 'reg_lambda': np.float64(1.6923737499050842), 'subsample': np.float64(0.9227651908203118)}
  Fold 2: RMSE=0.5655, Best Parameters={'colsample_bytree': np.float64(0.7367518666865607), 'gamma': np.float64(0.04589953290672094), 'learning_rate': np.float64(0.03824709648056803), 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 303, 'reg_alpha': np.float64(0.0017161101831750236), 'reg_lambda': np.float64(1.6923737499050842), 'subsample': np.float64(0.9227651908203118)}
  Fold 3: RMSE=0.5650, Best Parameters={'colsample_bytree': np.float64(0.7367518666865607), 'gamma': np.float64(0.04589953290672094), 'learning_rate': np.float64(0.03824709648056803), 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 303, 'reg_alpha': np.float64(0.0017161101831750236), 'reg_lambda': np.float64(1.6923737499050842), 'subsample': np.float64(0.9227651908203118)}
  Fold 4: RMSE=0.5661, Best Parameters={'colsample_bytree': np.float64(0.7367518666865607), 'gamma': np.float64(0.04589953290672094), 'learning_rate': np.float64(0.03824709648056803), 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 303, 'reg_alpha': np.float64(0.0017161101831750236), 'reg_lambda': np.float64(1.6923737499050842), 'subsample': np.float64(0.9227651908203118)}

```

