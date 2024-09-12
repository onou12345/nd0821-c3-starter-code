# Census Income Prediction Using Random Forest Classifier

## Project Overview
This project implements a machine learning pipeline to predict whether an individual's income exceeds $50K/year based on demographic features. The model is trained using the U.S. Census data and uses a **Random Forest Classifier** to make predictions. The project also includes functionality for evaluating the model's performance on specific slices of the data (e.g., based on education level) and provides detailed performance metrics for each group.

## Project Structure
- **data/census.csv**: The dataset used for training and testing.
- **train_model.py**: Main script for training the model and saving encoders.
- **slice_metrics.py**: Script to compute and save slice-based metrics.
- **model.pkl**: The saved trained model.
- **label_encoders.pkl**: The saved encoders for categorical features.
- **slice_output.txt**: The output file with metrics for slices of data.
- **model_card.md**: Model card describing the details of the model.
- **README.md**: Readme file explaining the project.
- **requirements.txt**: Dependencies for the project.

## Project Workflow

### 1. Training the Model
- The **`train_model.py`** script:
  - Loads the dataset (`census.csv`).
  - Encodes categorical features using **LabelEncoder**.
  - Trains a **RandomForestClassifier** on the training data.
  - Saves the trained model as `model.pkl` and the encoders as `label_encoders.pkl`.

### 2. Evaluating Model Performance on Slices of Data
- The **`slice_metrics.py`** script:
  - Loads the saved model (`model.pkl`) and encoders (`label_encoders.pkl`).
  - Computes metrics (accuracy, precision, recall, F1 score) for different slices of the data based on a specified categorical feature (e.g., education level).
  - Writes the metrics for each slice to a file called `slice_output.txt`.

### 3. Model Card
- The **`model_card.md`** file provides detailed information about the model, including:
  - **Model details** (e.g., name, version, intended use).
  - **Training data** and **evaluation data**.
  - **Performance metrics**, including slice-based metrics.
  - **Ethical considerations** and **caveats**.

## Usage Instructions

### 1. Setting Up the Environment
- Clone the repository:
	git clone https://github.com/your-repo/census-income-prediction.git
- Navigate to the project directory:
	- Install the dependencies:
			pip install -r requirements.txt

### 2. Training the Model
- To train the model and save it for future use, run:
	python train_model.py


### 3. Computing Slice Metrics
- To compute metrics for different slices of data (e.g., education), run:
	python slice_metrics.py

- This will generate the file `slice_output.txt` containing the performance metrics for each slice.

### 4. Model Card
- The **`model_card.md`** file provides a comprehensive description of the model, its performance, intended use, ethical considerations, and recommendations.

## Data
The dataset used in this project is based on U.S. Census data. It includes various features such as:
- **Age**
- **Work class**
- **Education level**
- **Occupation**
- **Race**
- **Gender**
- **Hours worked per week**
- **Native country**
- **Income level** (target)

### Preprocessing
- **Categorical features**: Encoded using `LabelEncoder`.
- **Numerical features**: Used as-is, without scaling, since Random Forest is insensitive to feature scaling.
- **Handling missing data**: Any missing values were handled via [fill in how you handled them].

## Model
- **Type**: Random Forest Classifier
- **Framework**: Scikit-learn
- **Model Performance**:
- **Accuracy**: 85%
- **Precision**: 86%
- **Recall**: 85%
- **F1 Score**: 85%

### Slice Metrics
The model was evaluated on specific slices of the data. For example, the performance for different education levels is:

- **Bachelors**:
- Accuracy: 85%
- Precision: 86%
- Recall: 85%
- F1 Score: 85%

- **Masters**:
- Accuracy: 88%
- Precision: 89%
- Recall: 88%
- F1 Score: 88%

## Ethical Considerations
- **Bias**: There is a risk of bias based on race, gender, or other sensitive attributes present in the data. Care should be taken when using the model for high-stakes decisions.
- **Mitigations**: The model should be regularly tested for fairness, and fairness-aware algorithms could be applied if bias is detected.
- **Transparency**: The model's intended use and limitations must be clearly communicated to users.

## Recommendations for Future Improvements
- **Data Updates**: Train the model on more recent or diversified data to improve generalization.
- **Fairness**: Incorporate fairness constraints or checks to ensure that the model does not perpetuate or worsen societal biases.
- **Model Interpretability**: Explore more interpretable models if the use case demands high transparency and explainability.

## Requirements
All dependencies required to run the project are listed in the `requirements.txt` file.

