import requests
from deepdiff import DeepDiff

url = "http://0.0.0.0:9696/predict"

features = {
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

actual_response = requests.post(url, json=features).json()

expected_response = {'price': 1000}

diff = DeepDiff(actual_response, expected_response)
print(f'diff={diff}')

assert 'type_changes' not in diff
