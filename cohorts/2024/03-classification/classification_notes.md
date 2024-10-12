# Binary Classification (Logistic Regression)

* `g(xi) ≈ yi`
* `yi ∈ {0,1}`

## Churn

Churn rate is a measure of the proportion of individuals or items leaving a group over a specific period. In this lesson, it refers to the likelihood of a client continuing to purchase (yi near 0) or ceasing to be a client (yi near 1).

**Classes, functions, and methods:** 

```py
from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=len(df_test), random_state=1)
```

## Feature Importance

### Difference and Risk Ratio

* `global` = the total population of the feature
* `group` = a filtered subset of the feature's population

```py
global_mean = df_train_full.churn.mean()
```

1. Difference:
    * `global - group`
    * `difference < 0` -> more likely to churn.
    * `difference > 0` -> less likely to churn.
2. Risk ratio
    * `group / global`
    * `risk > 1` -> more likely to churn.
    * `risk < 1` -> less likely to churn.

### Mutual information

The ***mutual information*** of 2 random variables is a measure of the mutual dependence between them.

In Scikit-Learn, in the Metrics package, the `mutual_info_score` method allows us to input 2 features and it will output the mutual information score.

The score can be between `0` and `1`. The closest to `1`, the more important the feature is.

**Classes, functions, and methods:** 

```py
from sklearn.metrics import mutual_info_score

def mutual_info_term_score(series):
    return mutual_info_score(series, y_train)

mi = df_train[categorical].apply(mutual_info_term_score)
mi.sort_values(ascending=False)
```

### Correlation

The ***correlation coefficient*** measures the linear correlation between 2 sets of data -> ratio between the covariance of 2 variables and the product of their standard deviations `𝝈`. In other words, it's a normalized covariance.

* `r` (also sometimes `𝝆`) = PCC or Pearson Correlation Coeficient.
* The value of `r` is always in the interval `[-1 ,1]`.
* If `r` is negative, when one of the variables grows, the other one decreases.
* If `r` is possitive, when one of the variables grows, the other one does as well.
* Values between `|0.0|` and `|0.2|`, the correlation is very low and growth/decrease is very softly reflected on the other variable.
* Values between `|0.2|` and `|0.5|` show moderate correlation.
* Values between `|0.5|` and `|1.0|` show strong correlation.

**Classes, functions, and methods:** 

* `df[x].corr()` -  returns the correlation between numerical columns of same data frame.
* `df[x].corrwith(y)` - returns the correlation between x and y series. This is a function from pandas.
* `df[x].corr().abs().style.background_gradient(cmap='viridis')` - easier to identify

### One-Hot Encoding

One-Hot Encoding allows encoding categorical variables in numerical ones. This method represents each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise. 

**Classes, functions, and methods:** 

```py
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
dv.fit(train_dict)
X_train = dv.transform(train_dict)
```

* `df[x].to_dict(oriented='records')` - convert x series to dictionaries, oriented by rows. 
* `DictVectorizer().fit_transform(x)` - Scikit-Learn class for converting x dictionaries into a sparse matrix, and in this way doing the one-hot encoding. It does not affect the numerical variables. 
* `DictVectorizer().get_feature_names_out()` -  returns the names of the columns in the sparse matrix.  

## Logistic Regression

In Logistic Regression, the model `g(xi)` will return a number between the values `[0,1]`. We can understand this value as the ***probability*** of `xi` belonging to the "positive class"; if the value were `1` then it would belong to this class, but if it were `0` it would belong to the opposite class of our binary classification problem.

* `g(xi) = sigmoid(z)`
* `z = wo + w^T · xi = linear regression` 
* Logistic Regression is similar to Linear Regression except that we wrap the original formula inside a _sigmoid_ function. The sigmoid function always returns values between `0` and `1`.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

## Training the Model

**Classes, functions, and methods:** 

```py
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()
```

* `LogisticRegression().fit(x, y)` - Scikit-Learn class for calculating the logistic regression model. 
* `LogisticRegression().coef_[0]` - returns the coeffcients or weights of the LR model
* `LogisticRegression().intercept_[0]` - returns the bias or intercept of the LR model
* `LogisticRegression().predict[x]` - make predictions on the x dataset 
* `LogisticRegression().predict_proba[x]` - make predictions on the x dataset, and returns two columns with their probabilities for the two categories - soft predictions 

## Accuracy

Different from linear regression we do not use "RMSE" here. We can check the accuracy of the model by comparing the predictions with the target (in other words, the error of our predictions) and calculating the mean of the error array. Even if the comparison vector is made of Booleans, NumPy will automatically convert them to 1's and 0's and calculate the mean.

## Logistic Regression workflow recap

1. Prepare the data
    1. Download and read the data with pandas
    1. Look at the data
    1. Clean up the feature/column names
    1. Check if all the columns read correctly (correct types, no NaN's, convert categorical target into numerical, etc)
    1. Check if the target data needs any preparation
1. Set up the validation framework (splits) with scikit-learn
1. Exploratory Data Analysis
    1. Check missing values
    1. Look at the target variable
        * Look at the distribution; use `normalize` for ease.
    1. Look at numerical and categorical variables
    1. Analyze feature importance
        * Difference and risk ratio
        * Mutual information
        * Correlation
1. Encode categorical features in one-hot vectors
1. Train the model with Logistic Regression
    1. Keep the prediction probabilities rather than the hard predictions if you plan on modifying the thresholds.
    1. Calculate the accuracy of the model with the validation dataset.
1. Interpret the model
    1. Look at the coefficients
    1. Train a smaller model with fewer features
1. Use the model
    * Combine the train and validation datasets for your final model and test it with the test dataset.