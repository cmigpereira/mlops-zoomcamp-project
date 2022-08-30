import os

import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

import mlflow

# MLFlow specific
TRACKING_SERVER = os.getenv("TRACKING_SERVER", "http://127.0.0.1:5000")
MODEL_NAME = "vehicles-price-model"
MLFLOW_FEATURES_DTYPES = 'models/features_dtypes.pkl'
MLFLOW_MODEL = 'models/model.xgb'

# where to store MLFLow artifacts locally
STORAGE_PATH = "artifacts"

# connect to MLFlow
mlflow.set_tracking_uri(TRACKING_SERVER)
client = MlflowClient(tracking_uri=TRACKING_SERVER)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/production")


def load_data():
    '''
    Loads XGBoost model, features values and types from MLFlow into memory
    '''

    path_exists = os.path.exists(STORAGE_PATH)

    if not path_exists:
        try:
            os.mkdir(STORAGE_PATH)
        except OSError as error:
            print(error)

    client.download_artifacts(
        run_id=model._model_meta.run_id, path="models", dst_path=STORAGE_PATH
    )

    features_dtypes = pd.read_pickle(f'{STORAGE_PATH}/{MLFLOW_FEATURES_DTYPES}')

    model_xgb = xgb.Booster()
    model_xgb.load_model(f'{STORAGE_PATH}/{MLFLOW_MODEL}')

    return model_xgb, features_dtypes


model_xgb, features_dtypes = load_data()


def predict_price(record):
    '''
    Creates proper XGB DMatrix and returns prediction capped at minimum of 0
    '''
    df = pd.json_normalize(record)

    # set type as used for training
    for col in df.columns:
        df[col] = df[col].astype(features_dtypes[col])

    df = xgb.DMatrix(data=df, enable_categorical=True)

    pred = model_xgb.predict(df)

    return max(0, int(round(pred[0])))


app = Flask('price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    record = request.get_json()

    pred = predict_price(record)

    result = {'price': pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
