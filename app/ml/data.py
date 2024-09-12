import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """

    df = pd.read_csv(file_path)
    # Mapping from input field names to the feature names used in training
    feature_name_mapping = {
        ' capital-gain': ' capital_gain',
        ' capital-loss': ' capital_loss',
        ' education-num': ' education_num',
        ' fnlgt': ' fnlwgt',
        ' hours-per-week': 'hours_per_week',
        ' marital-status': 'marital_status',
        ' native-country': 'native_country'
    }

    # Rename the columns in the DataFrame to match the model's feature names
    df.rename(columns=feature_name_mapping, inplace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return df


def encoder_helper(df, encoders=None, save_path=None):
    """
    Encode categorical features using LabelEncoder. If encoders are provided,
    use them, otherwise fit new encoders and save them.

    Parameters:
    df (DataFrame): DataFrame containing the categorical columns.
    encoders (dict, optional): Dictionary of pre-fitted encoders.

    Returns:
    DataFrame: DataFrame with encoded categorical columns.
    dict: Dictionary of encoders used for the encoding process.
    """
    # If no encoders are provided, create a new dictionary to store
    # LabelEncoders

    category_lst = df.select_dtypes(
        include=['object']).columns.to_list()
    if encoders is None:
        encoders = {}

    # Iterate through the list of categorical columns
    for category in category_lst:
        if category in encoders:
            # Use the provided encoder
            le = encoders[category]
            df[category] = le.transform(df[category])
        else:
            # Create and fit a new LabelEncoder
            le = LabelEncoder()
            df[category] = le.fit_transform(df[category])
            encoders[category] = le

    # Save the encoders if new ones were created
    if save_path:
        joblib.dump(encoders, save_path)

    return df
