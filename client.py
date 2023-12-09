import requests

url = "http://localhost:5000/predict"
res = requests.post(url, json={"text": "bad movie"})
print(res.text)