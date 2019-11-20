import itertools
from typing import List, DefaultDict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

from .categorical_encoders import LeaveOneOutEncoder


class LOOGridSearchCV:
    """
    Specially prepared class to do grid search with cross-validation on our loo encoded
    DataFrame.

    Scores should be approximately ok, although i have no proof for that :)
    """

    def __init__(self,
                 train_df: pd.DataFrame,
                 model,
                 params_grid: DefaultDict,
                 columns_to_encode: List,
                 columns_to_drop_from_training: List,
                 Xs_train: List[pd.DataFrame] = None,
                 ys_train: List[pd.DataFrame] = None,
                 Xs_val: List[pd.DataFrame] = None,
                 ys_val: List[pd.DataFrame] = None,
                 ohe_emails: bool = True,
                 mean: int = 1,
                 std: int = 0.05,
                 n_folds: int = 5,
                 encoded_df: pd.DataFrame = pd.DataFrame(),
                 ) -> None:
        """
        :param train_df: train_df (will be splitted then to train/and_val n_folds times)
        :param model: model to train
        :param params_grid: param_grid to search
        :param columns_to_encode: categorical columns, which you want to encode using loo
        :param columns_to_drop_from_training: columns to drop from training phase
        :param ohe_emails: if set to True, performs OHE on emails column
        :param Xs_train:
        :param mean: mean to regularization part of the encoding
        :param std: std to regularization part of the encoding
        :param n_folds: n_folds to validate
        :param encoded_df: if task was done before, just pass here already encoded_df
        "
        """

        self.processed_train_df = train_df.copy(deep=True)
        self.model = model
        self.params_grid = params_grid
        self.columns_to_encode = columns_to_encode
        self.columns_to_drop_from_training = columns_to_drop_from_training
        self.ohe_emails = ohe_emails
        self.mean = mean
        self.std = std
        self.n_folds = n_folds
        if not Xs_train:
            self.Xs_train, self.ys_train, self.Xs_val, self.ys_val = ([] for i in range(4))
        else:
            self.Xs_train = Xs_train
            self.ys_train = ys_train
            self.Xs_val = Xs_val
            self.ys_val = ys_val
        self.encoded_df_ = encoded_df

    def _ohe_emails(self) -> pd.DataFrame:
        """
        internal method for one hot encoding emails column
        """
        email_ohe_names = {0: '0_emails',
                           1: '1_email',
                           2: '2_emails',
                           3: '3_emails',
                           4: '4_emails',
                           5: '5_emails'}

        self.processed_train_df = (pd.concat([self.processed_train_df, pd.get_dummies(
            self.processed_train_df['emails'])], axis=1)
                                   .rename(columns=email_ohe_names))
        self.columns_to_drop_from_training.append('emails')

        return self.processed_train_df.copy(deep=True)

    def _prepare_train_val_dfs(self):
        """
        Internal method

        bunch of code to prepare train and validation dataframes for given n_folds, to make
        grid search and cross validation processes much faster: for each n_folds you will need
        to compute encoded_df only once, same for validation and train DataFrames
        """

        X = self.processed_train_df.reset_index().drop(columns='name')
        y = self.processed_train_df[['target']].reset_index().drop(columns='name')

        if self.ohe_emails:
            self.processed_train_df = self._ohe_emails()

        X.drop(columns=self.columns_to_drop_from_training, inplace=True)

        """
        to have each sample exactly once in validation set
        """
        kf = KFold(n_splits=self.n_folds, shuffle=False, random_state=None)
        splits = kf.split(self.processed_train_df)

        dfs_to_mean = []
        for train_index, val_index in splits:
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[val_index], y.iloc[val_index]
            X_val.drop(columns=['target'], inplace=True)

            enc = LeaveOneOutEncoder(train_df=X_train,
                                     test_df=X_val,
                                     columns_to_encode=self.columns_to_encode,
                                     target_column='target',
                                     random_state=42,
                                     mean=self.mean,
                                     std=self.std)

            X_train, X_val = enc.fit()
            encoded_cols = [col for col in X_train.columns if 'encoded_' in col]
            dfs_to_mean.append(X_train[encoded_cols])
            train_to_drop = self.columns_to_encode.copy()
            train_to_drop.extend(['target'])
            X_train.drop(columns=train_to_drop, inplace=True)
            test_to_drop = self.columns_to_encode.copy()
            X_val.drop(columns=test_to_drop, inplace=True)

            self.Xs_train.append(X_train)
            self.ys_train.append(y_train)
            self.Xs_val.append(X_val)
            self.ys_val.append(y_val)

        """
        we are computing here the mean of the folds with excluding the 'i am now validation not the 
        training set' part, as I see it as the most proper thing to do, to use cross-validation 
        approach
        """
        for df in dfs_to_mean:
            zeros = [0 for col in df.columns]
            for index in range(len(self.processed_train_df)):
                if index not in df.index:
                    df.loc[index, :] = zeros
            df.sort_index(inplace=True)

        mean_df = dfs_to_mean[0].copy(deep=True)
        mean_df = mean_df * 0
        for num in range(self.n_folds):
            mean_df = mean_df + dfs_to_mean[num]
        self.encoded_df_ = mean_df.divide(self.n_folds - 1)

    def grid_search(self) -> Tuple[List, List, List, List]:
        """
        performs GridSearchCV

        :return: list with each of the models: accuracies, parameters, recalls and confusion
        matrices
        """
        if self.encoded_df_.empty:
            self._prepare_train_val_dfs()

        models_accuracies, models_recalls, models_parameters, models_cms = ([] for i in range(4))
        for p in itertools.product(*self.params_grid.values()):
            model_params = self.params_grid.copy()
            for counter, key in enumerate(model_params.keys()):
                model_params[key] = p[counter]

            models_parameters.append(model_params.items())
            clf = self.model.set_params(**model_params)

            cv_accuracies, cv_recalls, cv_cms = ([] for i in range(3))

            """
            fitting and predicting for all folds, then scoring them by: 
            accuracy, recall and confusion matrix
            """
            for index in range(self.n_folds):
                clf.fit(self.Xs_train[index], self.ys_train[index])
                predictions = clf.predict(self.Xs_val[index])
                cv_accuracies.append(accuracy_score(self.ys_val[index], predictions))
                cv_recalls.append(recall_score(self.ys_val[index], predictions))
                cv_cms.append(confusion_matrix(self.ys_val[index], predictions))

            """
            final evaluation of scores (means of all folds scores
            for confusion matrix we can get not integer values, please treat this more informative
            than strict - but anyway, as a source of information which model should we choose
            """
            models_accuracies.append(np.mean(cv_accuracies))
            models_recalls.append(np.mean(cv_recalls))
            models_cms.append(np.mean(cv_cms, axis=0))

        return models_accuracies, models_parameters, models_recalls, models_cms

    def processed_train(self):
        """
        :return: processed train DataFrame with added encoded columns
        """
        train = self.processed_train_df.copy(deep=True)
        encoded = self.encoded_df_.copy(deep=True)
        train = train.reset_index().drop(columns=['name'])
        train = train.drop(columns=self.columns_to_drop_from_training+self.columns_to_encode)
        processed_train = pd.concat([train, encoded], axis=1)

        return processed_train
