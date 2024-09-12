from sklearn.model_selection import train_test_split
from pathlib import Path
from ml.data import load_data, encoder_helper
from ml.model import (
    train_model,
    model_inference,
    classification_metrics,
    save_model
)


if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir / '../data/census.csv'
    target_column = 'salary'

    df = load_data(data_path)

    df = encoder_helper(
        df,
        save_path=script_dir /
        '../model/label_encoders.pkl')
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    save_model(model, script_dir / "../model/model.pkl")

    y_pred = model_inference(model, X_test)
    metrics = classification_metrics(y_test, y_pred)
    print(metrics)
