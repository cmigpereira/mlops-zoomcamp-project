import os
import time
import datetime

import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

import mlflow

print('Loading configs')
TRACKING_SERVER = os.getenv("TRACKING_SERVER", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "vehicles-price-experiment"
MODEL_NAME = "vehicles-price-model"

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME")
DATASET_PATH = 'datasets'
TRAIN_SET_PQ = 'train.pq'
VAL_SET_PQ = 'val.pq'
TEST_SET_PQ = 'test.pq'

FEATURES_VALUES_PATH = "./features_values.pkl"
FEATURES_DTYPES_PATH = "./features_dtypes.pkl"

mlflow.set_tracking_uri(TRACKING_SERVER)

client = MlflowClient(TRACKING_SERVER)


@dag(
    schedule_interval=None, start_date=days_ago(1), catchup=False, tags=['vehicle-price']
)
def initial_deploy():  # pylint: disable=too-many-arguments
    '''
    ETL pipeline for Airflow
    '''

    def save_feature_values(path):
        '''
        Store possible features values in a pickle for uploading it to MLFlow
        '''
        print('Saving features values locally')
        train = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{TRAIN_SET_PQ}')
        val = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{VAL_SET_PQ}')
        test = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{TEST_SET_PQ}')

        df = pd.concat([train, val, test], ignore_index=True, sort=False)

        pd.Series({c: df[c].unique() for c in df}).to_pickle(path)

    def save_dtypes(df, path):
        '''
        Store possible dtypes in a pickle for uploading it to MLFlow
        '''
        print('Saving dtypes locally')
        dtypes = (
            df.dtypes.to_frame('dtypes')
            .reset_index()
            .set_index('index')['dtypes']
            .astype(str)
        )
        dtypes.to_pickle(path)

    def train_model(train, val):
        '''
        Train a model using a train and validation datasets, using MLFlow for experiment tracking
        '''
        print('Training model')
        df_train = train.copy()
        df_val = val.copy()

        target = 'price'
        y_train = df_train[target].values
        y_val = df_val[target].values

        X_train = df_train.drop(target, axis=1)
        X_val = df_val.drop(target, axis=1)

        cat_feats = X_train.select_dtypes(include=['object']).columns.to_list()

        for cat_feat in cat_feats:
            X_train[cat_feat] = X_train[cat_feat].astype('category')
            X_val[cat_feat] = X_val[cat_feat].astype('category')

        save_dtypes(X_train, FEATURES_DTYPES_PATH)

        train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        valid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        save_feature_values(FEATURES_VALUES_PATH)

        mlflow.autolog(disable=True)

        def objective(params):
            with mlflow.start_run(nested=True):
                mlflow.set_tag("model", "xgboost")
                mlflow.log_params(params)
                booster = xgb.train(
                    params=params,
                    dtrain=train,
                    num_boost_round=10,  # 300
                    evals=[(valid, 'validation')],
                    early_stopping_rounds=3,  # 50
                )
                y_pred = booster.predict(valid)
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                mlflow.log_metric("rmse", rmse)
                mlflow.xgboost.log_model(booster, artifact_path="models")
                mlflow.log_artifact(
                    local_path=FEATURES_VALUES_PATH, artifact_path="models"
                )
                mlflow.log_artifact(
                    local_path=FEATURES_DTYPES_PATH, artifact_path="models"
                )

            return {'loss': rmse, 'status': STATUS_OK}

        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42,
        }

        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=5,  # 50
            trials=Trials(),
        )

    def wait_until_ready(model_name, model_version):
        '''
        Waits max of 20 seconds until the model becomes Ready
        '''
        for _ in range(20):
            model_version_details = client.get_model_version(
                name=model_name,
                version=model_version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f'Model status: {ModelVersionStatus.to_string(status)}')
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)

    def register_model(experiment_name):
        '''
        Register model and set it to production
        '''
        print('Registering model')
        experiment = mlflow.get_experiment_by_name(experiment_name)
        best_run_df = mlflow.search_runs(
            [experiment.experiment_id], order_by=["metrics.rmse DESC"]
        )
        best_run_id = best_run_df.loc[0, 'run_id']

        model_details = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/models", name=MODEL_NAME
        )

        client.update_registered_model(
            name=model_details.name,
            description="This model forecasts the price of vehicles.",
        )

        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage='production',
            archive_existing_versions=True,
        )

        # wait until the model is ready
        wait_until_ready(model_details.name, model_details.version)

    @task
    def initial_model():
        '''
        Trains and registers the first model, including setting it to production stage
        '''
        print('Training initial model')
        train = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{TRAIN_SET_PQ}')
        val = pd.read_parquet(f'{AIRFLOW_HOME}/{DATASET_PATH}/{VAL_SET_PQ}')

        str_time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        exp_name = EXPERIMENT_NAME + '_' + str_time_now
        mlflow.set_experiment(exp_name)

        train_model(train, val)

        register_model(exp_name)

    initial_model()


initial_deploy_dag = initial_deploy()
