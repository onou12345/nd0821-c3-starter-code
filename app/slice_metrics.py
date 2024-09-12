import joblib
from train_model import load_data, load_model, encoder_helper
from train_model import model_inference, classification_metrics
from pathlib import Path


def load_encoders(encoder_path):
    """
    Load the saved label encoders using joblib.

    Parameters:
    encoder_path (str): Path to the saved encoders file.

    Returns:
    dict: Dictionary of LabelEncoders.
    """
    return joblib.load(encoder_path)


def compute_slice_metrics(model, X, y, categorical_column, original_df):
    """
    Compute model metrics for each slice of data based on the unique values
    of a given categorical column, and return the metrics with the
    original categorical values.
    Parameters:

    model: Trained model used for inference.
    X (DataFrame): Feature set for inference.
    y (Series): True labels.
    categorical_column (str): The name of the categorical column
    to slice the data by.
    original_df (DataFrame): The original DataFrame with
    unencoded categorical values.

    Returns:
    list: List of dictionaries containing metrics for each slice.
    """
    metrics_per_slice = []
    # Use original categorical values
    unique_values = original_df[categorical_column].unique()

    for value in unique_values:
        # Create a slice of the data using the original (pre-encoded)
        # categorical value
        slice_X = X[original_df[categorical_column] == value]
        slice_y = y[original_df[categorical_column] == value]

        # Make predictions for this slice
        slice_predictions = model_inference(model, slice_X)

        # Calculate metrics for the slice
        metrics = classification_metrics(slice_y, slice_predictions)

        # Append metrics to the list with the original categorical value
        metrics_per_slice.append({
            'category': categorical_column,
            'value': value,  # Use the original value
            'metrics': metrics
        })

    return metrics_per_slice


if __name__ == "__main__":
    # Load the dataset
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir / '../data/census.csv'
    df = load_data(data_path)
    original_df = df.copy()

    # Load the saved model and encoders
    model = load_model(script_dir / '../model/model.pkl')
    encoders = load_encoders(script_dir / '../model/label_encoders.pkl')

    # Get the categorical columns and apply encoders
    categorical_columns = df.select_dtypes(
        include=['object']).columns.to_list()
    df = encoder_helper(df, encoders=encoders)

    # Split data into features and target
    target_column = 'salary'
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Choose the categorical column to compute slice metrics (e.g.,
    # 'education')
    categorical_column = 'education'

    # Compute metrics for each slice of the chosen column
    slice_metrics = compute_slice_metrics(
        model, X, y, categorical_column, original_df)

    # Write the slice metrics to a file
    with open(script_dir / '../model/slice_output.txt', 'w') as f:
        for metric in slice_metrics:
            f.write(
                f"Category: {metric['category']}, Value: {metric['value']}\n")
            for m, value in metric['metrics'].items():
                f.write(f"{m}: {value}\n")
            f.write("\n")

    print("Metrics for each slice have been written to 'slice_output.txt'")
