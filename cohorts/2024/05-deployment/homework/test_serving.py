import requests
import json

url = 'http://localhost:9696/predict'
# customer = {"job": "student", "duration": 280, "poutcome": "failure"}
customer = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=customer)
result = response.json()

print(json.dumps(result, indent=2))