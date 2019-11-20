from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:
    """
    helps us with managing our raw/train/test DataFrames:
    facilitates us:
    - load raw data,
    - create train and test DataFrames,
    - read train and test DataFrames
    """

    def __init__(self,
                 local_path: Path = Path('/Users/mjasiecz/PycharmProjects/'),
                 project_path: Path = Path('new_offer_success_predictor/data/raw/'),
                 filename: str = 'client_database',
                 suffix: str = '.parquet',
                 csv_suffix: str = '.csv') -> None:
        """
        :param local_path: path to the local folder, you probably will need to change at least
         mjasiecz part
        :param project_path: path to project raw data
        :param filename: name of the file with raw data
        :param suffix: suffix of the file with raw data
        :param csv_suffix: .csv suffix
        """

        self.local_path = local_path
        self.project_path = project_path
        self.filename = filename
        self.suffix = suffix
        self.csv_suffix = csv_suffix

    def load_data(self) -> pd.DataFrame:

        data_path = (self.local_path
                     .joinpath(self.project_path)
                     .joinpath(Path(self.filename))
                     .with_suffix(self.suffix))

        df = pd.read_parquet(data_path, engine='pyarrow')
        # deletes empty (and not useful) rows from DataFrame
        df = df[df['accepted'].notna()].set_index('name')

        return df

    def _train_test_paths(self) -> Tuple[Path, Path]:
        """
        internal method for creating paths for train and test DataFrames
        """

        train_test_paths = {df_type: self.local_path
                                         .joinpath(self.project_path)
                                         .joinpath(df_type)
                                         .with_suffix(self.csv_suffix)
                            for df_type in ['train', 'test']}

        return train_test_paths['train'], train_test_paths['test']

    def create_train_test(self,
                          df: pd.DataFrame = pd.DataFrame()) -> None:
        """
        Splits raw DataFrame to train/test DataFrames.

        :param df: df with the raw data
        """

        if df.empty:
            df = self.load_data()

        train_dataset, test_dataset = self._train_test_paths()

        if train_dataset.exists() and test_dataset.exists():
            print('Split is already done. Do not data snoop!')
        else:
            print('Preparing train and test DataFrames.')
            df_predictors = df.drop(columns=['accepted'])
            df_target = df['accepted']
            (df_train,
             df_test,
             df_train_target,
             df_test_target) = train_test_split(df_predictors,
                                                df_target,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=df_target)
            df_train.insert(0, column='accepted', value=df_train_target)
            df_test.insert(0, column='accepted', value=df_test_target)
            df_train.to_csv(path_or_buf=train_dataset)
            df_test.to_csv(path_or_buf=test_dataset)
            print('DataFrames are ready to use.')

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads train and test DataFrames
        """

        train_dataset, test_dataset = self._train_test_paths()
        cond = train_dataset.exists() and test_dataset.exists()

        None if cond else self.create_train_test()

        df_train = pd.read_csv(train_dataset, index_col='name')
        df_test = pd.read_csv(test_dataset, index_col='name')

        return df_train, df_test

