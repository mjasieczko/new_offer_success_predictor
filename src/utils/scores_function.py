import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)


def scores_function(model,
                    X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame,
                    mode: str = 'loo') -> None:
    """
    function to print all relevant scores of our model

    :param model: ml_model
    :param X_train: train df
    :param X_test: test df
    :param y_train: train target labels
    :param y_test: test target labels
    :param mode: 'loo' for model from loo GridSearchCV
                 'pickle' for pickled, final model
                 'skl' for model from sklearn GridSearchCV
    """
    if mode == 'loo':
        roc_train = roc_auc_score(y_train, model.best_roc_auc_estimator.predict(X_train))
        predictions = model.best_roc_auc_estimator.predict(X_test)
    if mode == 'pickle':
        roc_train = roc_auc_score(y_train, model.predict(X_train))
        predictions = model.predict(X_test)
    if mode == 'skl':
        roc_train = roc_auc_score(y_train, model.best_estimator_.predict(X_train))
        predictions = model.best_estimator_.predict(X_test)

    roc_auc = roc_auc_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    c_m = confusion_matrix(y_test, predictions)

    print(f'generalization error: {roc_train - roc_auc}')
    print(f'roc_auc_score: {roc_auc}')
    print(f'recall_score: {recall}')
    print(f'accuracy_score: {accuracy}')
    print(f'precision_score: {precision}')
    print('confusion_matrix: ')
    print()
    print(c_m)
