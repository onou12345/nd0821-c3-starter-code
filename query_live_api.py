import requests

# Replace with your actual Heroku app URL
url = "https://census-salary-prediction-2ba1424bb86e.herokuapp.com/predict/"

# Define the input data that you want to send to the API
input_data = {
    "age": 25,
    "workclass": "Private",
    "fnlwgt": 226802,
    "education": "11th",
    "education_num": 7,
    "marital_status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Send a POST request to the /predict/ endpoint
response = requests.post(url, json=input_data)

# Print the status code and raw response content
print(f"Status Code: {response.status_code}")
print(f"Raw Response: {response.text}")

# If the status code is 200 (Success), attempt to parse the JSON
if response.status_code == 200:
    try:
        print(f"Response JSON: {response.json()}")
    except ValueError:
        print("Response is not in JSON format")
else:
    print(f"Failed with status code: {response.status_code}")
