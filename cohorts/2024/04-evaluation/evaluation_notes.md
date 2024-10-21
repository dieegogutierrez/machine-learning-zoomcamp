## Accuracy, Dummy Model and Class Imbalance

Accuracy measures the fraction of correct predictions. Specifically, it is the number of correct predictions divided by the total number of predictions.

A first approach to improving our accuracy would be to change the ***classification threshold***. We used `0.5` as a threshold before but a different threshold value may lead to better results.

By moving the threshold to both extremes (`threshold = 0` and `threshold = 1`) and training different models between both extremes, we can create ***dummy models*** for which we can calculate the accuracy and calculate which threshold value has the highest accuracy.

The threshold at both extremes can tell us interesting info about our model. In the _churn database_ example, a threshold of `1` turns out to have an accuracy of 73%; the maximum accuracy was with threshold `0.5` and turned out to be 80%, which is better, but not by a large margin.

By analyzing the dataset we find that it has severe ***class imbalance***, because the proportion of `churn` clients to `no_churn` is about `3:1`.

Therefore, the accuracy metric in cases with class imbalance is misleading and does not tell us how well our model performs compared to a dummy model.

**Classes and methods:**

* `np.linspace(x,y,z)` - returns a numpy array starting at `x` until `y` with `z` evenly spaced samples 
* `Counter(x)` - collection class that counts the number of instances that satisfy the `x` condition
* `accuracy_score(x, y)` - sklearn.metrics class for calculating the accuracy of a model, given a predicted `x` dataset and a target `y` dataset. 

```py
from sklearn.metrics import accuracy_score
from collections import Counter

accuracy_score(y_val, y_pred >= 0.5)
thresholds = np.linspace(0, 1, 21)
Counter(y_pred >= 1.0)
```

## Confusion table

For binary classification, based on the prediction and the ground truth, there are 4 posible outcome scenarios:
* Ground truth positive, prediction positive > Correct > ***True positive***
* Ground truth positive, prediction negative > Incorrect > ***False positive***
* Ground truth negative, prediction posiive > Incorrect > ***False negative***
* Ground truth negative, prediction negative > Correct > ***True negative***

The ***confusion table*** is a matrix whose columns (x dimension) are the predictions and the rows (y dimension) is the ground truth:

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2"><b>Predictions</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Actual</b></td>
      <td><b>Negative</b></td>
      <td><b>Positive</b></td>
    </tr>
   <tr>
      <td><b>Negative</b></td>
      <td>TN</td>
      <td>FP</td>
    </tr>
    <tr>
      <td><b>Positive</b></td>
      <td>FN</td>
      <td>TP</td>
    </tr>
  </tbody>
</table>

Each position contains the element count for each scenario. We can also convert the count values to percentages.

## Precision and Recall

**Precision** tell us the fraction of positive predictions that are correct. It takes into account only the **positive class** (TP and FP - second column of the confusion matrix), as is stated in the following formula:


$$P = \cfrac{TP}{TP + FP}$$


**Recall** measures the fraction of correctly identified postive instances. It considers parts of the **postive and negative classes** (TP and FN - second row of confusion table). The formula of this metric is presented below: 


$$R = \cfrac{TP}{TP + FN}$$


**Accuracy** in the churn example model, the accuracy is of 80% but the precision drops to 67% and the recall is only 54%. Thus, the 80% accuracy value is very misleading. This is caused by class imbalance.

$$A = \cfrac{TP + TN}{TP + TN + FP + FN}$$

## ROC curves

***ROC*** stands for ***Receiver Operating Characteristic***.

We begin by defining the ***False Positive Rate*** and ***True Positive Rate***:

* `FPR = FP / (TN + FP)`
* `TPR = TP / (TP + FN)`

Note that `TPR = recall`.

We want the FPR to be as low and TPR to be as high as possible in any model.

If we try different thresholds and calculate confusion tables for each threshold, we can also calculate the TPR and FPR for each threshold.

When we plot the FPR (x axis) against the TPR (y axis), a random baseline model should describe an ascending straight diagonal line, a perfect model would increase inmediately to 1 and stay up, and our model most likely will be somewhere in between in a bow shape, ascending quickly at first and then decreasing the growth until it reaches the point (1,1), almost asymptotically.

**Classes and methods:** 
* `np.repeat([x,y], [z,w])` - returns a numpy array with a z number of x values, and a w number of y values. 
* `roc_curve(x, y)` - sklearn.metrics class for calculating the false positive rates, true positive rates, and thresholds, given a target x dataset and a predicted y dataset. 

## ROC AUC

The Area under the ROC curves can tell us how good is our model with a single value. The AUROC of a random model is 0.5, while for an ideal one is 1. 

In other words, AUC can be interpreted as the probability that a randomly selected positive example has a greater score than a randomly selected negative example.

**Classes and methods:** 

* `auc(x, y)` - sklearn.metrics class for calculating area under the curve of the x and y datasets. For ROC curves x would be false positive rate, and y true positive rate. 
* `roc_auc_score(x, y)` - sklearn.metrics class for calculating area under the ROC curves of the x false positive rate and y true positive rate datasets. x = true, y = predict probabilities.
* `randint(x, y, size=z)` - np.random class for generating random integers from the “discrete uniform”; from `x` (inclusive) to `y` (exclusive) of size `z`. 

## Cross Vailidation - K FOLD

**Cross-validations** refers to evaluating the same model on different subsets of a dataset, getting the average prediction, and spread within predictions. This method is applied in the **parameter tuning** step, which is the process of selecting the best parameter.

In this algorithm, the full training dataset is divided into **k partitions**, we train the model in k-1 partitions of this dataset and evaluate it on the remaining subset. Then, we end up evaluating the model in all the k folds, and we calculate the average evaluation metric for all the folds.

**Libraries, classes and methods:**

- `Kfold(k, s, x)` - sklearn.model_selection class for calculating the cross validation with k folds, s boolean attribute for shuffle decision, and an x random state
- `Kfold.split(x)` - sklearn.Kfold method for splitting the x dataset with the attributes established in the Kfold's object construction.
- `for i in tqdm()` - library for showing the progress of each i iteration in a for loop.