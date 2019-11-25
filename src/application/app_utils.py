import pickle
import warnings
from pathlib import Path

import pandas as pd

from data.data_manager import DataManager
from data.data_processor import DataProcessor, TestDataProcessor
from ml_preprocessing.categorical_encoders import LeaveOneOutEncoder

warnings.filterwarnings('ignore')


def run(arg_unseen_data_path: Path, arg_output_path: Path):
    """
    main script of new_offer_success_predictor repo

    predicts offer acceptance probabilities (probabilities of success i.e. customer will
    accept our offer) using the best overall model (via roc_auc, recall, accuracy and precision
     metrics)

     saves results to excel file (xlsx) in form:
     customer_name | success_probability
        1                   x
        2                   y
    etc.
     """
    """
    read train data to help encode test set
    """
    DM = DataManager()
    train_df = DM.load_data()

    """
    firefighting
    """
    arg_unseen_data_path = Path(Path(str(arg_unseen_data_path).split('.')[0]))

    """
    read unseen data to predict their class
    """
    DM_unseen = DataManager(local_path=Path(arg_unseen_data_path),
                            project_path=Path(''),
                            filename='',
                            suffix='.parquet',
                            csv_suffix='.csv')
    unseen_df = DM_unseen.load_data()

    """
    only for testing reasons
    """
    if 'accepted' in unseen_df.columns:
        unseen_df = unseen_df.drop(columns=['accepted'])

    """
    process both train and test data
    """
    customer_names = unseen_df.reset_index()[['name']].rename(columns={'name': 'customer_name'})
    DP = DataProcessor(train_df=train_df)
    processed_train_df = DP.perform_initial_features_engineering()
    TDP = TestDataProcessor(not_processed_train_df=train_df,
                            processed_train_df=processed_train_df,
                            test_df=unseen_df,
                            sneaky_peaky=True)
    processed_unseen_df = TDP.perform_initial_features_engineering()

    columns_to_encode = ['offer_class',
                         'gender',
                         'customer_type',
                         'center',
                         'phone_calls',
                         'cc_len',
                         'cc_startswith']
    """
    encoding test set
    """
    enc = LeaveOneOutEncoder(train_df=processed_train_df,
                             test_df=processed_unseen_df,
                             columns_to_encode=columns_to_encode,
                             target_column='target',
                             random_state=42,
                             mean=1,
                             std=0.05)
    _, test_df_encoded = enc.fit()
    test_df_encoded_ohemails = test_df_encoded.copy(deep=True)

    """
    dictionary for email ohe mapping
    """
    email_ohe_names = {0: '0_emails',
                       1: '1_email',
                       2: '2_emails',
                       3: '3_emails',
                       4: '4_emails',
                       5: '5_emails'}

    test_df_encoded_ohemails = (
        pd.concat([test_df_encoded_ohemails, pd.get_dummies(test_df_encoded_ohemails['emails'])],
                  axis=1).rename(columns=email_ohe_names)).drop(columns=['emails'])

    """
    features used to predict on test set
    """
    test_columns = ['log_salary', 'log_estimated_expenses_knn', 'log_offer_value_knn',
                    'nan_age', 'not_nan_age', '0_emails', '1_email', '2_emails',
                    '3_emails', '4_emails', '5_emails', 'encoded_offer_class',
                    'encoded_gender', 'encoded_customer_type', 'encoded_center',
                    'encoded_phone_calls', 'encoded_cc_len', 'encoded_cc_startswith']
    unseen_data = test_df_encoded_ohemails[test_columns]

    """
    load model
    """
    models_path = (Path('/Users/mjasiecz/PycharmProjects/new_offer_success_predictor/models/final_model.pickle'))
    if not models_path.exists():
        print('be sure to change model path my friend :)')
    final_model = pickle.load(open(models_path, 'rb'))

    """
    predict probability
    """
    probabilities = pd.DataFrame(
        {'success_probability': final_model.predict_proba(unseen_data)[:, 1]}
    )
    result = pd.merge(customer_names,
                      probabilities,
                      how='inner',
                      on=customer_names.index).drop(columns='key_0')

    """
    generate results
    """
    result.to_excel(arg_output_path,
                    sheet_name='cust_prob_list',
                    engine='xlsxwriter')

    print('results were generated to: '+str(arg_output_path))
