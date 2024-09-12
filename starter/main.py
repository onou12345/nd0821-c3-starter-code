from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from app.ml.data import encoder_helper
from pathlib import Path

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model and encoders
script_dir = Path(__file__).parent.absolute()
model = joblib.load(script_dir / "model/model.pkl")
encoders = joblib.load(script_dir / "model/label_encoders.pkl")

label_mapping = {
    0: "<=50K",
    1: ">50K"
}

# Define the input schema using Pydantic


class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Create the GET endpoint


@app.get("/")
async def root():
    return {"message": "Welcome to the model inference API!"}

# Helper function to encode input data


def encode_input(input_data: InputData):
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    df = encoder_helper(df, encoders)
    return df

# Create the POST endpoint for inference


@app.post("/predict/")
async def predict(data: InputData):
    processed_data = encode_input(data)
    # Perform inference
    prediction = model.predict(processed_data)

    # Fallback to "Unknown" if something goes wrong
    salary_label = label_mapping.get(prediction[0], "Unknown")

    # Return the string label for the salary prediction
    return {"Predicted Salary": salary_label}
