# Decision Trees

```py
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_val)[:, 1]
```

## Tunning

* `max_depth` (default: `None`).
* `min_samples_leaf` (default: `1`).

## Useful

```py
from sklearn.tree import export_text
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))
```

# Random Forest

- Multiple Decision Trees in parallel.

```py
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
```

## Tunning

- Besides `max_depth` and `min_samples_leaf`, Random forests have also:

* `n_estimators` (default: `100`). Number of trees in the forest.
* `max_features` (default: `auto`). Number of features to consider for each split.
* `bootstrap` (default: `True`). If `False`, the whole dataset is used to build each tree. If `True`, each tree is build with random subsamples with replacement (datapoints can be repeated), AKA bootstrapping.

# XGBoost

- Multiple Decision Trees in sequential line. A new decision tree is created with the errors of the previous.

```bash
pip install xgboost
```
```py
import xgboost as xgb
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

%%capture output # Jupyter command to capture the output that later will be placed on a dataframe.

xgb_params = {
    'eta': 0.3, # Learn rate
    'max_depth': 6, # Depth of the tree
    'min_child_weight': 1, # Same as min_samples_leaf
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1, # Type of warnings that will show
}

watchlist = [(dtrain, 'train'), (dval, 'val')] # Used to evaluate the model while in process of training
model = xgb.train(xgb_params, dtrain, num_boost_round=10, evals=watchlist, verbose_eval=5) # num_boost_round is the number of trees. verbose_eval the number of steps to be printed

y_pred = model.predict(dval)
```

- Function for extracting the xgb output into a dataframe:
```py
def parse_xgb_output(output):
    results = []
    
    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')
        
        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])
        
        results.append((it, train, val))
    
    columns = ['n_iter', 'train_rmse', 'val_rmse']
    df_results = pd.DataFrame(results, columns=columns)
    
    return df_results
```

## Tunning

In XGBoost there are 3 hyperparameters of importance:

* `eta` ( 𝜂 , default: `0.3`), AKA ***learning rate***. Defines the weight applied to the new predictions in order to correct the previous predictions (in other words, the step size of each optimization). Bigger learning rate leads to faster training but worse accuracy because it cannot finetune the results; smaller learning rate results in more accurate results but takes longer to train.
* `max_depth` (default: `6`); virtually the same as in scikit-learn.
* `min_child_weight` (default: `1`); virtually the same as `min_samples_leaf` in scikit-learn.

Other hyperparameters of interest:

* `colsample_bytree` (default: `1`) is the subsample ratio of features/columns when constructing each tree.
* `subsample` (default: `1`) is the subsample ratio of the training instances/rows.
* `lambda` (default: `1`) AKA L2 regularization.
* `alpha` (default: `0`) AKA L1 regularization.