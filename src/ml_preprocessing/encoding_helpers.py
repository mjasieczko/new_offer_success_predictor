import itertools
from typing import Tuple, List, DefaultDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

from .categorical_encoders import LeaveOneOutEncoder


def loo_grid_search_cv(train_df,
                       model: str,
                       params_grid: DefaultDict,
                       columns_to_encode: List,
                       columns_to_drop_from_training: List,
                       email_ohe_names: DefaultDict = None,
                       mean: int = 1,
                       std: int = 0.05,
                       n_folds: int = 5) -> Tuple[pd.DataFrame, List, List, List, List]:
    """
    Specially prepared function to do grid search with cross-validation on our loo encoded
    DataFrame.

    Scores should be approximately ok, although i have no proof for that :)

    :param train_df: train_df (will be splitted then to train/and_val n_folds times)
    :param model: model to train, currently supports only LogisticRegression
                  and RandomForestClassifier
    :param params_grid: param_grid to search
    :param columns_to_encode: categorical columns, which you want to encode using loo
    :param columns_to_drop_from_training: columns to drop from training phase
    :param email_ohe_names: Dictionary containing names for new columns of One Hot Encoded emails
    :param mean: mean to regularization part of the encoding
    :param std: std to regularization part of the encoding
    :param n_folds: n_folds to validate
    :return: mean encoded DataFrame, scores for each model: accuracy, parameters, recalls,
             confusion matrices
    """

    processed_train_df = train_df.copy(deep=True)

    X = processed_train_df.reset_index().drop(columns='name')
    y = (processed_train_df[['target']]
         .reset_index()
         .drop(columns='name'))

    """
    one hot encoding of email column
    """
    if email_ohe_names:
        processed_train_df = (pd.concat([processed_train_df,
                                         pd.get_dummies(processed_train_df['emails'])], axis=1)
                                .rename(columns=email_ohe_names))
        columns_to_drop_from_training.append('emails')

    X.drop(columns=columns_to_drop_from_training, inplace=True)

    """
    what we do here is preparing training/validation datasets, then encoding them, then we
    pass this information further to appropriate modelling part
    (we do the split here, because it is sufficient to do categorical encoding once here, than
    doing it over and over in the modelling part)
    """
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=None)
    """
    to have each sample exactly once in validation set
    """

    splits = kf.split(processed_train_df)

    dfs_to_mean = []
    Xs_train, ys_train, Xs_val, ys_val = ([] for i in range(4))
    for train_index, val_index in splits:
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]
        X_val.drop(columns=['target'], inplace=True)

        enc = LeaveOneOutEncoder(train_df=X_train,
                                 test_df=X_val,
                                 columns_to_encode=columns_to_encode,
                                 target_column='target',
                                 random_state=42,
                                 mean=mean,
                                 std=std)
        X_train, X_val = enc.fit()
        encoded_cols = [col for col in X_train.columns if 'encoded_' in col]
        dfs_to_mean.append(X_train[encoded_cols])
        train_to_drop = columns_to_encode.copy()
        train_to_drop.extend(['target'])
        X_train.drop(columns=train_to_drop, inplace=True)
        test_to_drop = columns_to_encode.copy()
        X_val.drop(columns=test_to_drop, inplace=True)

        Xs_train.append(X_train)
        ys_train.append(y_train)
        Xs_val.append(X_val)
        ys_val.append(y_val)

    """
    we are computing here the mean of the folds with excluding the 'i am now validation not the 
    training set' part, as I see it as the most proper thing to do, to use cross-validation 
    approach
    """
    for df in dfs_to_mean:
        zeros = [0 for col in df.columns]
        for index in range(len(processed_train_df)):
            if index not in df.index:
                df.loc[index, :] = zeros
        df.sort_index(inplace=True)

    mean_df = dfs_to_mean[0].copy(deep=True)
    mean_df = mean_df * 0
    for num in range(n_folds):
        mean_df = mean_df + dfs_to_mean[num]
    encoded_final_df = mean_df.divide(n_folds - 1)

    """
    GridSearch right here
    """
    models_accuracies, models_recalls, models_parameters, models_cms = ([] for i in range(4))
    for p in itertools.product(*params_grid.values()):
        model_params = params_grid.copy()
        for counter, key in enumerate(model_params.keys()):
            model_params[key] = p[counter]

        models_parameters.append(model_params.items())
        if model == 'LogisticRegression':
            clf = LogisticRegression(**model_params)

        if model == 'RandomForestClassifier':
            clf = RandomForestClassifier(**model_params)

        cv_accuracies, cv_recalls, cv_cms = ([] for i in range(3))

        """
        fitting and predicting for all folds, then scoring them by: 
        accuracy, recall and confusion matrix
        """
        for index in range(n_folds):
            clf.fit(Xs_train[index], ys_train[index])
            predictions = clf.predict(Xs_val[index])
            cv_accuracies.append(accuracy_score(ys_val[index], predictions))
            cv_recalls.append(recall_score(ys_val[index], predictions))
            cv_cms.append(confusion_matrix(ys_val[index], predictions))

        """
        final evaluation of scores (means of all folds scores
        for confusion matrix we can get not integer values, please treat this more informative
        than strict - but anyway, as a source of information which model should we choose
        """
        models_accuracies.append(np.mean(cv_accuracies))
        models_recalls.append(np.mean(cv_recalls))
        models_cms.append(np.mean(cv_cms, axis=0))

    return encoded_final_df, models_accuracies, models_parameters, models_recalls, models_cms
