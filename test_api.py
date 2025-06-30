import requests

# Simulate patient data input (5 values)
data = {
    "features": [0.3, 1.2, 0.5, 2.1, 1.0]
}

# Send a POST request to the Flask server
response = requests.post("http://127.0.0.1:5000/predict", json=data)

# Show the server's response
print(response.json())



