import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    processes data, is well adapted for processing training DataFrame,
    - deals with missing data,
    - converts column types for further work (CategoricalEncoding of categorical columns),
    - performs initial features engineering
    """

    def __init__(self,
                 train_df: pd.DataFrame) -> None:
        """
        :param train_df: training DataFrame
        """
        self.df = train_df.copy(deep=True)

    def missing_information(self,
                            percentage: bool = False) -> pd.Series:
        """
        shows us information about missing data

        :param percentage: set to True, if you want to have percentage information on the output
        :return: Series with information on missing rows in each column
        """
        df_shape = self.df.shape
        print(f'df.shape: {df_shape}')
        is_null_sum = self.df.isnull().sum()
        missing_information = (is_null_sum
                               if not percentage
                               else (is_null_sum / df_shape[0] * 100).round(2))
        return missing_information

    @staticmethod
    def _group_columns() -> defaultdict:
        """
        returns dictionary with logically grouped columns
        """
        groups = defaultdict()
        groups['age'] = ['age']
        groups['emails_and_phone_calls'] = ['emails', 'phone_calls']
        groups['low_cardinality'] = ['offer_class', 'gender', 'customer_type', 'center']
        groups['high_cardinality'] = ['customer_code', 'offer_code', 'number']
        groups['numerical'] = ['salary', 'offer_value', 'estimated_expenses']
        groups['target'] = ['accepted']
        return groups

    @staticmethod
    def _missing_salary(df: pd.DataFrame) -> pd.Series:
        """
        fills missing data in salary column:
        for people with the same customer_code fills the same data as in other row,
        for other people
        """
        df = df.copy(deep=True)
        df['salary_temp'] = df['salary'].fillna(0.1)
        mask = ['customer_type', 'customer_code', 'salary_temp']
        temp = (df[mask]
                .groupby(mask)
                .size()
                .reset_index()
                .rename(columns={0: 'size'}))
        temp = (temp[temp
                .duplicated(subset=['customer_code', 'customer_type'],
                            keep=False)][['customer_code', 'salary_temp']])
        temp = temp[temp['salary_temp'] != 0.1]
        temp = temp.set_index('customer_code')['salary_temp']
        salary_map = temp.to_dict()
        df['salary'] = (df['salary']
                        .fillna(df['customer_code']
                                .map(salary_map)))
        temp = (df[['customer_type', 'salary']]
                .groupby('customer_type')
                .median())
        temp = temp['salary']
        salary_map = temp.to_dict()
        df['salary'] = (df['salary']
                        .fillna(df['customer_type']
                                .map(salary_map)))
        # to be able to np.log('salary')
        df.loc[df['salary'] == 0, 'salary'] = 0.0001
        return df['salary']

    def _prepare_to_knn(self,
                        df: pd.DataFrame) -> pd.DataFrame:
        """
        prepares data for knn imputing (of missing data)
        for training data uses also labels to give more information then to test set
        """
        df = df.copy(deep=True)
        temp = pd.get_dummies(df['gender'])
        groups = self._group_columns()
        if 'accepted' in df.columns:
            temp = pd.concat([pd.get_dummies(df['accepted']), temp], axis=1)
        for col in itertools.chain(groups['emails_and_phone_calls'], groups['low_cardinality']):
            temp = pd.concat([pd.get_dummies(df[col]), temp], axis=1)
        knn_to_complete = pd.concat([temp, df[groups['numerical'] + groups['age']]], axis=1)
        return knn_to_complete

    def deal_with_missing_values(self,
                                 n_neighbors: int = 5) -> pd.DataFrame:
        """
        :param n_neighbors: number of neighbours for knn algorithm
        :return: returns DataFrame with filled nans (also, leaves unchanged 'age' column ->
        to be used further for possible age binning)
        """
        df = self.df.copy(deep=True)
        groups = self._group_columns()
        for col in groups['high_cardinality']:
            df[col] = df[col].fillna('missing')

        def _prt_process_emails_and_phone_calls(
                emails_phone_calls_df: pd.DataFrame) -> pd.DataFrame:
            """
            initially processes emails and phone_calls columns:
            models some outliers with almost full no responses, to have constant value
            """
            emails_phone_calls_df.loc[emails_phone_calls_df['emails'] > 4, 'emails'] = 5
            emails_phone_calls_df.loc[emails_phone_calls_df['phone_calls'] > 3, 'phone_calls'] = 4
            return emails_phone_calls_df

        for col in groups['emails_and_phone_calls']:
            df[col] = df[col].fillna(round(df[col].mean()))

        df = _prt_process_emails_and_phone_calls(df)

        for col in groups['low_cardinality']:
            df[col] = df[col].fillna(method='ffill')
        df['salary'] = self._missing_salary(df)
        temp = (df[groups['emails_and_phone_calls']
                   + groups['age']
                   + groups['low_cardinality']
                   + groups['numerical']])
        knn_unfilled_table = self._prepare_to_knn(temp)
        knn_filled = (KNN(k=n_neighbors,
                          print_interval=1032)
                      .fit_transform(knn_unfilled_table
                                     .to_numpy()))
        knn_imputed_cols = ['age_knn', 'estimated_expenses_knn', 'offer_value_knn']
        for col in knn_imputed_cols:
            df[col] = knn_filled[:, -knn_imputed_cols.index(col) - 1]
        df = df.drop(columns=['offer_value', 'estimated_expenses'])
        return df

    @staticmethod
    def _process_age(age_df: pd.DataFrame) -> pd.DataFrame:
        """
        Divides age for 2 bins: where age is nan and opposite case
        (we consider situation, when it's possible that client does not gave us their age because
        he wasn't truly interested in cooperation with us)
        """
        age_df = age_df.copy(deep=True)
        age_df['nan_age'] = age_df['age'].isna()
        age_df['not_nan_age'] = age_df['age'].notna()
        return age_df[['nan_age', 'not_nan_age']]

    @staticmethod
    def _process_target(target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps target:
        1 for accepted
        0 for not accepted
        (for ML algorithms purposes)
        """
        target_df = target_df.copy(deep=True)
        target_df['target'] = target_df['accepted'] == 'yes'
        return target_df[['target']]

    @staticmethod
    def _process_high_cardinality_categorical_cols(high_cardinal_df: pd.DataFrame) -> pd.DataFrame:
        """
        creates features for high cardinality categorical columns: (those which won't bring any
        additional value where encoded)
        at this moment we consider only customer_code column as profitable
        """
        high_cardinal_df = high_cardinal_df.copy(deep=True)
        high_cardinal_df['cc_len'] = high_cardinal_df['customer_code'].str.len()
        high_cardinal_df.loc[high_cardinal_df['cc_len'].isin([5, 8]), 'cc_len'] = '58'
        high_cardinal_df.loc[~high_cardinal_df['cc_len'].isin(['58']), 'cc_len'] = 'ELSE'
        high_cardinal_df['cc_startswith'] = high_cardinal_df['customer_code']
        a_p_c = ['A', 'P', 'C']
        for letter in a_p_c:
            (high_cardinal_df
                .loc[high_cardinal_df['customer_code']
                                      .str
                                      .startswith(letter), 'cc_startswith']) = letter
        (high_cardinal_df
            .loc[~high_cardinal_df['customer_code']
                                   .str
                                   .startswith(tuple(a_p_c)), 'cc_startswith']) = 'ELSE'
        return high_cardinal_df[['cc_len', 'cc_startswith']]

    @staticmethod
    def _process_numerical_cols(numerical_df: pd.DataFrame) -> pd.DataFrame:
        """
        creates features:
        - standard scales numerical columns,
        - log scales numerical columns,
        to consider three cases when modelling:
        only standard scaled, only log scaled, and mixed using variances
        """
        numeric_cols = [col for col in numerical_df.columns if numerical_df[col].dtype != object]
        numeric_cols.remove('age')
        temp = numerical_df[numeric_cols]
        scaler = StandardScaler()
        for col in numeric_cols:
            temp['log_' + col] = np.log(temp[col])
        log_cols = [col for col in temp.columns if col not in numerical_df.columns]
        log_subset = temp[log_cols]
        temp_subset = temp[numeric_cols]
        scaled = scaler.fit_transform(temp_subset)
        scaled = pd.DataFrame(scaled,
                              columns='scaled_' + temp_subset.columns,
                              index=temp.index)
        processed = pd.concat([log_subset, scaled], axis=1)
        return processed

    def perform_initial_features_engineering(self) -> pd.DataFrame:
        """
        performs initial feature engineering (without encoding -> will be done as individual part
        due to some maintenance issues (how to cross validate target encoding?)
        """
        groups = self._group_columns()
        df = self.deal_with_missing_values().copy(deep=True)
        for col in groups['emails_and_phone_calls']:
            df[col] = df[col].astype(object)
        age = self._process_age(df)
        numerical = self._process_numerical_cols(df)
        high_cardinal = self._process_high_cardinality_categorical_cols(df)
        columns_to_drop = ['customer_code', 'number', 'offer_code']
        if 'accepted' in df.columns:
            columns_to_drop.append('accepted')
            target = self._process_target(df)
            df = pd.concat([target, df], axis=1)
        df = pd.concat([age, df], axis=1)
        df = pd.concat([numerical, df], axis=1)
        df = pd.concat([high_cardinal, df], axis=1)
        df.drop(columns=columns_to_drop, inplace=True)

        return df


class TestDataProcessor(DataProcessor):
    """
    DataProcessor adapted to test set needs
    """

    def __init__(self,
                 not_processed_train_df,
                 processed_train_df,
                 test_df,
                 sneaky_peaky=True) -> None:
        """
        :param not_processed_train_df: self explanatory
        :param processed_train_df: self explanatory
        :param test_df: self explanatory
        :param sneaky_peaky: set to True:
            uses some knn columns as 'original' to bring a little 'overfitting' to test set.
            As we have only lot of missing values in age, tries to sneak some correlation between
            age and responses
        """
        DataProcessor.__init__(self, not_processed_train_df)

        if sneaky_peaky:
            self.df['age'] = processed_train_df['age_knn']
            self.df['estimated_expenses'] = processed_train_df['estimated_expenses_knn']
            self.df['offer_value'] = processed_train_df['offer_value_knn']
        self.df = (pd
                   .concat([self
                           .df
                           .drop(columns=['salary', 'accepted']), processed_train_df['salary']],
                           axis=1))

        self.train_len = len(processed_train_df)
        self.df = pd.concat([self.df, test_df], axis=0).copy(deep=True)

    def perform_initial_features_engineering(self) -> pd.DataFrame:
        """
        performs initial features engineering on test set
        """
        df = DataProcessor.perform_initial_features_engineering(self)
        df = df[self.train_len:]

        return df







