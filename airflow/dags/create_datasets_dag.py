import os
from datetime import timedelta

import kaggle
import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split

# where to send reference file for monitoring
EVIDENTLY_SERVICE_PATH = os.getenv("EVIDENTLY_SERVICE_PATH", "monitor")

# Kaggle credentials
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
# Kaggle URL dataset
DATASET_KAGGLE_URL = 'austinreese/craigslist-carstrucks-data'

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME")
DATASET_PATH = 'datasets'
DATASET = 'vehicles.csv'
PREPROCESS_DATASET_PQ = 'vehicles_preprocessed.pq'
TRAIN_DATASET_PQ = 'train.pq'
VAL_DATASET_PQ = 'val.pq'
TEST_DATASET_PQ = 'test.pq'


@dag(
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=10),
    tags=['vehicle-price'],
)
def create_datasets():
    @task
    def download_dataset():
        '''
        Get dataset from Kaggle API
        '''
        print('Downloading Dataset')
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_file(
                DATASET_KAGGLE_URL,
                file_name=DATASET,
                path=f'{AIRFLOW_HOME}/{DATASET_PATH}',
            )
        except kaggle.api.rest.ApiException as exception:
            print(exception)

    def load_download_dataset():
        '''
        Load downloaded dataset
        '''
        df = pd.read_csv(f'{AIRFLOW_HOME}/{DATASET_PATH}/{DATASET}.zip')

        return df

    @task()
    def pre_process():
        '''
        Pre-process dataset
        '''
        print('Accessing Dataset')
        df = load_download_dataset()
        print('Pre-process Dataset')
        # convert posting_date to time
        df['posting_date'] = pd.to_datetime(
            df['posting_date'], format='%Y/%m/%dT%H:%M:%S', utc=True, errors='coerce'
        ).dt.tz_convert(None)
        df['posting_date'] = pd.to_datetime(df['posting_date'])
        df = df.sort_values(by=['id', 'posting_date'], ascending=True)
        print(df.dtypes)
        # dropping irrelevant columns for modelling as well as the ones
        # with large number of missing values
        df = df.drop(
            [
                'id',
                'url',
                'region_url',
                'condition',
                'cylinders',
                'title_status',
                'VIN',
                'image_url',
                'description',
                'county',
                'lat',
                'long',
                'size',
            ],
            axis=1,
        )
        df = df.dropna(
            subset=[
                'year',
                'manufacturer',
                'fuel',
                'odometer',
                'transmission',
                'posting_date',
            ]
        )
        # fill NAs
        df = df.fillna('Unknown')
        # add variable for counting days since first post
        FIRST_POSTING_DATE = pd.Timestamp('2021-04-01')
        df['days_since_202104'] = (df['posting_date'] - FIRST_POSTING_DATE).dt.days
        df = df.drop(['posting_date'], axis=1)
        df.to_parquet(
            f'{AIRFLOW_HOME}/{DATASET_PATH}/{PREPROCESS_DATASET_PQ}', index=False
        )

    @task()
    def split_save_dataset():
        '''
        Split dataset in train, val, test datasets
        '''
        df = load_parquet_dataset()
        print('Split Dataset')
        df_train, df_test = train_test_split(df, test_size=0.15, shuffle=False)
        df_train, df_val = train_test_split(df_train, test_size=0.15, shuffle=False)

        save_datasets(df_train, df_val, df_test)
        # reference file to evidently
        save_reference_file(df_train)

    def load_parquet_dataset():
        '''
        Load dataset
        '''

        df = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{PREPROCESS_DATASET_PQ}')

        return df

    def save_datasets(df_train, df_val, df_test):
        '''
        Save datasets to folder
        '''
        print('Save Datasets')
        df_train.to_parquet(
            f'{AIRFLOW_HOME}/{DATASET_PATH}/{TRAIN_DATASET_PQ}', index=False
        )
        df_val.to_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{VAL_DATASET_PQ}', index=False)
        df_test.to_parquet(
            f'{AIRFLOW_HOME}/{DATASET_PATH}/{TEST_DATASET_PQ}', index=False
        )

    def save_reference_file(df_train):
        '''
        Save dataset to folder
        '''
        print('Save reference file')
        df_train.to_parquet(
            f'{AIRFLOW_HOME}/{EVIDENTLY_SERVICE_PATH}/{TRAIN_DATASET_PQ}',
            index=False,
        )

    (
        download_dataset() >> pre_process() >> split_save_dataset()
    )  # pylint: disable=expression-not-assigned


create_datasets_dag = create_datasets()
