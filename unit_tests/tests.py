from pathlib import Path

import pandas as pd
import xgboost as xgb

# get local directory for pytest
test_directory = Path(__file__).parent
# locations for model info
STORAGE_PATH = "model"
MLFLOW_FEATURES_DTYPES = 'features_dtypes.pkl'
MLFLOW_FEATURES_VALUES = 'features_values.pkl'
MLFLOW_MODEL = 'model.xgb'


def test_load_data():
    '''
    Loads XGBoost model, features values and types from MLFlow into memory and
    checks if features dtypes has 12 columns
    '''
    try:
        model_xgb = xgb.Booster()
        model_xgb.load_model(f'{test_directory}/{STORAGE_PATH}/{MLFLOW_MODEL}')
        features_dtypes = pd.read_pickle(
            f'{test_directory}/{STORAGE_PATH}/{MLFLOW_FEATURES_DTYPES}'
        )
        features_values = pd.read_pickle(
            f'{test_directory}/{STORAGE_PATH}/{MLFLOW_FEATURES_VALUES}'
        )
    except OSError as e:
        print(e.errno)

    assert len(features_dtypes) == 12, "Missing dtypes"
    assert len(features_values) == 13, "Missing possible values"


def test_predict_price():
    '''
    Loads XGBoost model, features values and types from MLFlow into memory
    '''
    model_xgb = xgb.Booster()
    model_xgb.load_model(f'{test_directory}/{STORAGE_PATH}/{MLFLOW_MODEL}')
    features_dtypes = pd.read_pickle(
        f'{test_directory}/{STORAGE_PATH}/{MLFLOW_FEATURES_DTYPES}'
    )
    record = {
        "region": "SF bay area",
        "year": 1900.0,
        "manufacturer": "acura",
        "model": "\"t\"",
        "fuel": "diesel",
        "odometer": 0.0,
        "transmission": "automatic",
        "drive": '4wd',
        "type": "SUV",
        "paint_color": "Unknown",
        "state": "ak",
        "days_since_202104": 507,
    }

    df = pd.json_normalize(record)

    # set type as used for training
    for col in df.columns:
        df[col] = df[col].astype(features_dtypes[col])

    df = xgb.DMatrix(data=df, enable_categorical=True)

    pred = model_xgb.predict(df)

    assert isinstance(pred.item(), (int, float)), "Wrong pred type"
    assert (pred.item() >= 1000) & (pred.item() <= 1000000), "Pred outside normal values"
