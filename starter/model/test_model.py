import os
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from train_model import load_data, train_model, save_model, model_inference, load_model, encoder_helper, classification_metrics

# Set the path to the actual data file
data_path = '../data/census.csv'
target_column = ' salary'

def test_load_data():
    """
    Test the load_data function to ensure it loads a CSV file correctly 
    and returns a DataFrame that is not empty and contains the expected columns.
    """
    # Test load_data function
    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert target_column in df.columns

def test_encoder_helper():
    """
    Test the encoder_helper function to ensure that categorical columns are 
    correctly encoded using LabelEncoder and that the encoders are saved.
    """
    # Load the actual data
    df = load_data(data_path)
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    
    # Test encoder helper
    df_encoded = encoder_helper(df.copy(), categorical_columns)
    
    # Check if the categorical columns are transformed
    for col in categorical_columns:
        assert df_encoded[col].dtype == 'int64'
    
    # Check if the encoders file was saved
    assert os.path.exists('label_encoders.pkl')
    
    # Clean up
    os.remove('label_encoders.pkl')

def test_train_model():
    """
    Test the train_model function to ensure it correctly trains a model 
    and the save_model function saves it to the specified path.
    """
    # Load and prepare data for training
    df = load_data(data_path)
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    df_encoded = encoder_helper(df.copy(), categorical_columns)
    
    y = df_encoded[target_column]
    X = df_encoded.drop(target_column, axis=1)
    
    # Train the model
    model = train_model(X, y)
    assert model is not None
    
    # Save model
    save_model(model, 'test_model.pkl')
    
    # Check if model was saved
    assert os.path.exists('test_model.pkl')
    
    # Clean up
    os.remove('test_model.pkl')

def test_model_inference():
    """
    Test the model_inference function to ensure that it returns predictions 
    with the correct data type and the expected length.
    """
    # Load and prepare data for training
    df = load_data(data_path)
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    df_encoded = encoder_helper(df.copy(), categorical_columns)
    
    y = df_encoded[target_column]
    X = df_encoded.drop(target_column, axis=1)
    
    # Train the model
    model = train_model(X, y)
    
    # Perform inference
    X_new = X.copy()
    predictions = model_inference(model, X_new)
    
    # Check if the predictions are of correct type and size
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_new)

def test_classification_metrics():
    """
    Test the classification_metrics function to ensure that it returns a dictionary 
    containing the accuracy, precision, recall, and F1 score for the predictions.
    """
    # Load and prepare data for training and testing
    df = load_data(data_path)
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    df_encoded = encoder_helper(df.copy(), categorical_columns)
    
    y = df_encoded[target_column]
    X = df_encoded.drop(target_column, axis=1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model and perform inference
    model = train_model(X_train, y_train)
    y_pred = model_inference(model, X_test)
    
    # Test classification metrics
    metrics = classification_metrics(y_test, y_pred)
    
    # Ensure the returned metrics contain the necessary keys
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
