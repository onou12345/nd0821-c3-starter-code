from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """
    Test the GET method on the root endpoint.
    This checks the status code and the contents of the response.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the model inference API!"}


def test_post_predict_salary_lessthan_50k():
    """
    Test the POST method with input that should return <=50K.
    This checks that the model predicts the correct salary category.
    """
    # Example input that should return <=50K
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

    response = client.post("/predict/", json=input_data)
    assert response.status_code == 200
    assert response.json() == {"Predicted Salary": "<=50K"}


def test_post_predict_salary_greaterthan_50k():
    """
    Test the POST method with input that should return >50K.
    This checks that the model predicts the correct salary category.
    """
    # Example input that should return >50K
    input_data = {
        "age": 50,
        "workclass": "Private",
        "fnlwgt": 456789,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 50000,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    }

    response = client.post("/predict/", json=input_data)
    assert response.status_code == 200
    assert response.json() == {"Predicted Salary": ">50K"}
