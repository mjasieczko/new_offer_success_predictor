from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(local_path: Path = Path('/Users/mjasiecz/PycharmProjects/'),
              project_path: Path = Path('new_offer_success_predictor/data/raw/'),
              filename: str = 'client_database',
              suffix: str = '.parquet') -> pd.DataFrame:
    """

    :param local_path: path to the local folder, you probably will need to change at least mjasiecz
    part
    :param project_path: path to project raw data
    :param filename: name of the file with raw data
    :param suffix: suffix of this file
    :return: loaded DataFrame with raw data
    """

    data_path = (local_path
                 .joinpath(project_path)
                 .joinpath(Path(filename))
                 .with_suffix(suffix))

    df = pd.read_parquet(data_path, engine='pyarrow')

    return df


def create_train_test(df: pd.DataFrame = None,
                      local_path: Path = Path('/Users/mjasiecz/PycharmProjects/'),
                      project_path: Path = Path('new_offer_success_predictor/data/raw/'),
                      csv_suffix: str = '.csv') -> None:
    """
    Splits DataFrame to train/test datasets.

    :param df: df with the raw data
    :param local_path: path to the local folder, you probably will need to change at least mjasiecz
    part
    :param project_path: path to project raw data
    :param csv_suffix: suffix of the resulting df's
    """

    if not df:
        df = load_data()

    train_dataset = (local_path
                     .joinpath(project_path)
                     .joinpath('train')
                     .with_suffix(csv_suffix))

    test_dataset = (local_path
                    .joinpath(project_path)
                    .joinpath('test')
                    .with_suffix(csv_suffix))

    if train_dataset.exists() and test_dataset.exists():
        print('Split is already done. Do not data snoop!')
    else:
        print('Preparing train and test datasets.')
        df = df[df['accepted'].notna()].set_index('name')
        df_predictors = df.drop(columns=['accepted'])
        df_target = df['accepted']
        df_train, df_test, df_train_target, df_test_target = train_test_split(df_predictors,
                                                                              df_target,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              stratify=df_target)
        df_train.insert(0, column='accepted', value=df_train_target)
        df_test.insert(0, column='accepted', value=df_test_target)
        df_train.to_csv(path_or_buf=train_dataset)
        df_test.to_csv(path_or_buf=test_dataset)
        print('Datasets are ready to use.')
