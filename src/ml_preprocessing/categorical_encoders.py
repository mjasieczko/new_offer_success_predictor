from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd


class CategoricalEncoders:
    """
    Superclass for all categorical encoders possibly implemented here
    """

    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 columns_to_encode: List[str],
                 target_column: str) -> None:
        """
        :param train_df: training DataFrame
        :param test_df:  testing DataFrame
        :param columns_to_encode: labels of the categorical columns to encode
        :param target_column: label of the target column
        """

        self.train_df = train_df
        self.test_df = test_df
        self.columns_to_encode = columns_to_encode
        self.target = target_column


class LeaveOneOutEncoder(CategoricalEncoders):
    """
    Simple implementation of leave one out encoding of categorical columns.
    Written for binary classification.
    Warning! you must deal with missing values on your own, before using this class.

    how it works?

    Example:

    TRAIN
        user	target	encoded_user
    0	a	    0	    0.683224
    1	a	    1	    0.331029
    2	a	    1	    0.344128
    3	a	    0	    0.717434

    TEST
        user	encoded_user
    0	a	    0.5
    1	a	    0.5

    look at wacax post for explanation, based on Owen Zhang idea.
    1) https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748
    2) https://datascience.stackexchange.com/questions/10839/
    what-is-difference-between-one-hot-encoding-and-leave-one-out-encoding
    """

    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 columns_to_encode: List[str],
                 target_column: str,
                 random_state: int,
                 mean: float,
                 std: float) -> None:
        """
        :param random_state: random state for normal distribution numbers generator
        :param mean: mean for normal distribution numbers generator
        :param std: std for normal distribution numbers generator
        """
        CategoricalEncoders.__init__(self,
                                     train_df=train_df,
                                     test_df=test_df,
                                     columns_to_encode=columns_to_encode,
                                     target_column=target_column)
        self.random_state = random_state
        self.mean = mean
        self.std = std

    @staticmethod
    def _make_groups(df: pd.DataFrame,
                     columns_to_encode: List[str],
                     target_column: str) -> defaultdict:
        """
        internal method for making groups table (how many people responsed positively/negatively
        inside of categories
        """
        groups = defaultdict()
        for col in columns_to_encode:
            mask = [col, target_column]
            temp_df = df[mask]
            groups[col] = (temp_df
                           .groupby(mask)
                           .size()
                           .reset_index()
                           .rename(columns={0: 'size'}))
        return groups

    @staticmethod
    def _test_encoding(row: str, groups: pd.DataFrame, col: str) -> float:
        """
        internal method for encoding test set (function to further use in pd.apply)
        """
        temp = groups[groups[col] == row]
        numerator = temp[temp['target'] == 1].reset_index().loc[:, 'size']
        denominator = temp['size'].sum()
        return numerator / denominator

    def _loo_train(self) -> pd.DataFrame:
        """
        internal method
        performs loo on train set (we have response (target) column)
        """
        df = self.train_df.copy(deep=True)
        groups = self._make_groups(df=self.train_df,
                                   columns_to_encode=self.columns_to_encode,
                                   target_column=self.target)
        for col in self.columns_to_encode:
            transformed_rows = []
            np.random.seed(seed=self.random_state)
            group = groups[col]

            for row in df.index:
                row = df.loc[row]
                target = row[self.target]
                column_class = row[col]
                mean_numerator = (
                    group[(group[col] == column_class) & (group[self.target] == 1)]['size']
                    .reset_index()
                    .loc[:, 'size']
                    .values[0]
                )
                if target:
                    mean_numerator -= 1
                mean_denominator = group.loc[group[col] == column_class]['size'].sum() - 1
                random_number = np.random.normal(loc=self.mean,
                                                 scale=self.std,
                                                 size=1)[0]
                mean_response = mean_numerator / mean_denominator
                transformed_row = mean_response * random_number
                transformed_rows.append(transformed_row)

            df['encoded_' + col] = transformed_rows
        return df

    def _loo_test(self) -> pd.DataFrame:
        """
        internal method
        performs loo on train set (we don't have response (target) column)
        """
        df = self.test_df.copy(deep=True)
        groups = self._make_groups(df=self.train_df,
                                   columns_to_encode=self.columns_to_encode,
                                   target_column=self.target)

        for col in self.columns_to_encode:
            df['encoded_' + col] = (df[col]
                                    .apply(lambda row: self._test_encoding(row, groups[col], col)))

        return df

    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: Returns DataFrames with encoded chosen categorical columns inside both train
        and test sets
        """
        fitted_train = self._loo_train()
        fitted_test = self._loo_test()
        return fitted_train, fitted_test
