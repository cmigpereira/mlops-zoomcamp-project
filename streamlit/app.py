import os
import json
import datetime

import pandas as pd
import requests
from PIL import Image
from mlflow.tracking import MlflowClient

import mlflow
import streamlit as st

# AWS env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_BUCKET = "mlflow-bucket-cpereira-mlops-zoomcamp-cpereira"

# MLFlow specific
TRACKING_SERVER = os.getenv("TRACKING_SERVER", "http://127.0.0.1:5000")
MODEL_NAME = "vehicles-price-model"
MLFLOW_FEATURES_VALUES = 'models/features_values.pkl'

# connect to MLFlow
mlflow.set_tracking_uri(TRACKING_SERVER)
client = MlflowClient(tracking_uri=TRACKING_SERVER)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/production")

# where to store MLFLow artifacts locally
STORAGE_PATH = "artifacts"

# Flask API
FLASK_API_ENDPOINT = os.getenv("FLASK_API_ENDPOINT", "http://127.0.0.1:9696/predict")

# layout of webpage
st.set_page_config(
    page_title='Vehicles Price',
    layout='centered',
    page_icon=None,
    initial_sidebar_state='auto',
)
# image for webpage
IMAGE_PATH = 'imgs/fiat-500.jpg'


@st.cache()
def load_data():
    '''
    Loads features values from MLFlow into memory
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

    features_values = pd.read_pickle(f'{STORAGE_PATH}/{MLFLOW_FEATURES_VALUES}')

    return features_values


features_values = load_data()


def predict(row, feats_col):
    '''
    Sends a prediction request to the Flask API endpoints and returns the predicted listed price
    '''
    # Convert the  image to a NumPy array
    df = pd.DataFrame([row], columns=feats_col)

    inputs = df.to_dict(orient='records')

    # Send to the API
    response = requests.post(FLASK_API_ENDPOINT, json=inputs)

    # Leave for now exception here
    if response.status_code == 200:  # pylint: disable=no-else-return
        return response.text
    else:
        raise Exception(f"Status: {response.status_code}")


def app():
    '''
    The actual app
    '''

    st.markdown(
        "<h2 style='text-align: center; color: black;'>Predict the value of your vehicle</h2>",
        unsafe_allow_html=True,
    )

    image = Image.open(IMAGE_PATH)

    st.markdown("""***""")
    st.image(image)
    st.markdown("""***""")

    # Get the input from user for prediction with a form
    with st.form("predict"):
        region = st.selectbox(
            "Region",
            (sorted(features_values['region'])),
            format_func=lambda x: 'Region' if x == '' else x,
        )
        year = st.selectbox(
            "Year",
            (sorted(features_values['year'])),
            format_func=lambda x: 'Year' if x == '' else x,
        )
        manufacturer = st.selectbox(
            "Manufacturer",
            (sorted(features_values['manufacturer'])),
            format_func=lambda x: 'Manufacturer' if x == '' else x,
        )
        model = st.selectbox(
            "Model",
            (sorted(features_values['model'])),
            format_func=lambda x: 'Model' if x == '' else x,
        )
        fuel = st.selectbox(
            "Fuel",
            (sorted(features_values['fuel'])),
            format_func=lambda x: 'Fuel' if x == '' else x,
        )
        odometer = st.selectbox(
            "Odometer",
            (sorted(features_values['odometer'])),
            format_func=lambda x: 'Odometer' if x == '' else x,
        )
        transmission = st.selectbox(
            "Transmission",
            (sorted(features_values['transmission'])),
            format_func=lambda x: 'Transmission' if x == '' else x,
        )
        drive = st.selectbox(
            "Drive",
            (sorted(features_values['drive'])),
            format_func=lambda x: 'Drive' if x == '' else x,
        )
        type = st.selectbox(
            "Type",
            (sorted(features_values['type'])),
            format_func=lambda x: 'Type' if x == '' else x,
        )
        paint_color = st.selectbox(
            "Paint color",
            (sorted(features_values['paint_color'])),
            format_func=lambda x: 'Paint color' if x == '' else x,
        )
        state = st.selectbox(
            "State",
            (sorted(features_values['state'])),
            format_func=lambda x: 'State' if x == '' else x,
        )
        days_since_202104 = (datetime.datetime.now() - pd.to_datetime('2021-04-01')).days

        feat_cols = [
            'region',
            'year',
            'manufacturer',
            'model',
            'fuel',
            'odometer',
            'transmission',
            'drive',
            'type',
            'paint_color',
            'state',
            'days_since_202104',
        ]

        row = [
            region,
            year,
            manufacturer,
            model,
            fuel,
            odometer,
            transmission,
            drive,
            type,
            paint_color,
            state,
            days_since_202104,
        ]

        st.text("\n")

        if st.form_submit_button('Predict Price'):
            result_json = json.loads(predict(row, feat_cols))
            result = result_json['price']

            st.write(f'The vehicle should be listed around: `{result}$`')


if __name__ == '__main__':
    app()
