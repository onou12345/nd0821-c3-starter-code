import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
    
def save_model(model, file_path):
    """
    Save the trained model to a file using joblib.

    Parameters:
    model: Trained model object.
    file_path (str): Path to save the model.
    """
    joblib.dump(model, file_path)
   
def model_inference(model, X_new):
    """
    Perform inference using the trained model on new data.

    Parameters:
    model: Trained model object.
    X_new (DataFrame): New input features for inference.

    Returns:
    array: Predicted labels.
    """
    return model.predict(X_new)
    
def load_model(file_path="model.pkl"):
    """
    Load a previously saved model from a file.

    Parameters:
    file_path (str): Path to the saved model file.

    Returns:
    Model: Loaded model object.
    """
    return joblib.load(file_path)

def encoder_helper(df, category_lst):
    """
    Encode categorical features using LabelEncoder and save the encoders.

    Parameters:
    df (DataFrame): DataFrame containing the categorical columns.
    category_lst (list): List of column names to be encoded.

    Returns:
    DataFrame: DataFrame with encoded categorical columns.
    """
    label_encoders = {}
    # Iterate through the list of categorical columns
    for category in category_lst:
        le = LabelEncoder()
        df[category] = le.fit_transform(df[category])
        label_encoders[category] = le 

    joblib.dump(label_encoders, 'label_encoders.pkl')

    return df

def classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics including accuracy, precision, recall, and F1 score.

    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.

    Returns:
    dict: Dictionary containing classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":

    data_path = '../data/census.csv'
    target_column = ' salary'
    
    df = load_data(data_path)
    
    categorical_columns = df.select_dtypes(include=['object']).columns.to_list()
    
    df = encoder_helper(df, categorical_columns)
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    
    save_model(model, "model.pkl")
    
    y_pred = model_inference(model, X_test)
    metrics = classification_metrics(y_test, y_pred)
    print(metrics)