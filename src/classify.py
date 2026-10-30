import numpy as np
import pandas as pd
import scipy
import copy

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, chi2
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True):
    """Top k Accuracy classification score.
    For multiclass classification tasks, this metric returns the
    number of times that the correct class was among the top k classes
    predicted.

    Parameters
    ----------
    y_true : 1d array-like, or class indicator array / sparse matrix
        shape num_samples or [num_samples, num_classes]
        Ground truth (correct) classes.

    y_pred : array-like, shape [num_samples, num_classes]
        For each sample, each row represents the
        likelihood of each possible class.
        The number of columns must be at least as large as the set of possible
        classes.
    k : int, optional (default=5) predictions are counted as correct if
        probability of correct class is in the top k classes.

    normalize : bool, optional (default=True)
        If ``False``, return the number of top k correctly classified samples.
        Otherwise, return the fraction of top k correctly classified samples.

    Returns
    -------
    score : float
        If ``normalize == True``, return the proportion of top k correctly
        classified samples, (float), else it returns the number of top k
        correctly classified samples (int.)

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also
    --------
    accuracy_score

    Notes
    -----
    If k = 1, the result will be the same as the accuracy_score (though see
    note below). If k is the same as the number of classes, this score will be
    perfect and meaningless.

    In cases where two or more classes are assigned equal likelihood, the
    result may be incorrect if one of those classes falls at the threshold, as
    one class must be chosen to be the nth class and the class chosen may not
    be the correct one.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_pred = np.array([[0.1, 0.3, 0.4, 0.2],
    ...                    [0.4, 0.3, 0.2, 0.1],
    ...                    [0.2, 0.3, 0.4, 0.1],
    ...                    [0.8, 0.1, 0.025, 0.075]])
    >>> y_true = np.array([2, 2, 2, 1])
    >>> top_k_accuracy_score(y_true, y_pred, k=1)
    0.5
    >>> top_k_accuracy_score(y_true, y_pred, k=2)
    0.75
    >>> top_k_accuracy_score(y_true, y_pred, k=3)
    1.0
    >>> top_k_accuracy_score(y_true, y_pred, k=2, normalize=False)
    3
    """
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter
    

def filter_cor(mat, cutoff=0.95, negate=False):
    '''filter highly correlated features.

    It only filter away one of every highly correlated pair (the one
    that has higher overall correlation with other features.

    Parameters
    ----------

    '''
    # only care about upper triangle
    high = np.triu(mat >= cutoff, 1)
    s = set()
    for i, j in zip(*np.where(high)):
        if i in s or j in s:
            continue
        sumi = np.sum(mat[i, :])
        sumj = np.sum(mat[:, j])
        if sumi > sumj:
            s.add(i)
        else:
            s.add(j)
    return s


def train_test_rf(train, test, yname, n):
    assert(np.all(train.feature_metadata.index == test.feature_metadata.index))
    ylabels = train.sample_metadata[yname].unique().tolist()
    le = preprocessing.LabelEncoder()
    le.fit(ylabels)
    y = le.transform(train.sample_metadata[yname].tolist())
    sel = SelectKBest(chi2, k=100)
    sel.fit(train.data, y)
    X = sel.transform(train.data)
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [ 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "criterion": ["gini", "entropy"]}
    # oob requires bootstrap is True
    rf = RandomForestClassifier(n_estimators=500, oob_score=True, bootstrap=True)
    best_score = 0
    for g in ParameterGrid(param_grid):
        rf.set_params(**g)
        rf.fit(X,y)
        if rf.oob_score_ > best_score:
            best_model = copy.deepcopy(rf)
            best_score = rf.oob_score_
            best_grid = g
    print('best parameters:', best_grid)
    print('best score:', best_score)
    yhatall = []
    for i in range(50):
        test_sub = test.subsample_count(n, inplace=False)
        X_test = sel.transform(test_sub.data)
        yhat = pd.DataFrame(best_model.predict_proba(X_test))
        yhat.columns = le.inverse_transform(range(len(ylabels)))
        yhat['sid_'] = test_sub.sample_metadata.index
        yhat['obs_'] = test_sub.sample_metadata[yname].values
        yhat['iteration_'] = i
        yhatall.append(yhat)
    yhatall = pd.concat(yhatall)
    yhatall['count_'] = n
    return yhatall
