from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score
import joblib
from sklearn.ensemble import RandomForestClassifier


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


def load_model(file_path):
    """
    Load a previously saved model from a file.

    Parameters:
    file_path (str): Path to the saved model file.

    Returns:
    Model: Loaded model object.
    """
    return joblib.load(file_path)


def classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics including accuracy,
    precision, recall, and F1 score.

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
        "f1_score": f1}
