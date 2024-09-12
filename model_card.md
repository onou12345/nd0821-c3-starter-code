# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Name: RandomForestClassifier for Census Income Prediction
Version: 1.0
Model Type: Random Forest Classifier
Developers: Anastasios Foliadis
Date: 12.09.2024
Framework: Scikit-learn (Python)
Purpose: This model predicts whether an individual's income exceeds $50K/year based on demographic features.
## Intended Use
Primary use case: The model is designed for classifying individuals as having an income above or below $50K per year, based on features such as age, education level, occupation, etc.
Intended Users: Policy makers, data analysts, and social researchers to analyze income trends across different demographics.
Applications: This model could be used for research on income inequality, demographic studies, or applications that need income prediction based on available census data.
Limitations:
This model is trained on census data and may not generalize well outside similar contexts.
It does not account for future socio-economic changes or fluctuations in income determinants.
## Training Data
Data Source: The model is trained on the U.S. Census dataset, which contains demographic information for individuals.
Size: 32561 rows, 15 features.
Data Fields: The dataset includes features such as age, work class, education, marital status, occupation, race, gender, hours per week, and native country, among others.
Preprocessing:
Categorical columns were label-encoded.
Numerical features were kept as is without scaling due to the tree-based model's insensitivity to scaling.
## Evaluation Data
Source: The test set consists of 20% of the original dataset split via stratified train-test splitting.
Preprocessing: The test data was processed similarly to the training data, with the same encoding and handling for missing values.
## Metrics
Metrics Used: Accuracy, Precision, Recall, F1 Score

Performance:

Accuracy: 85%
Precision: 86%
Recall: 85%
F1 Score: 85%
These metrics were calculated using the test data and are based on the model's performance in distinguishing between individuals earning above and below $50K/year.

Metrics on slices of data: The model's performance was also evaluated on slices of data for different demographic groups such as education levels:

Education: Bachelors:
Accuracy: 85%
Precision: 86%
Recall: 85%
F1 Score: 85%
Education: Masters:
Accuracy: 88%
Precision: 89%
Recall: 88%
F1 Score: 88%
## Ethical Considerations
Bias and Fairness: There is a potential for bias in the model because it is trained on demographic data, which may lead to discrimination based on gender, race, or other sensitive attributes.
Mitigations: The model should be monitored for biases, particularly with respect to race and gender. Use fairness-aware techniques if necessary to mitigate any such biases.
Transparency: The model’s use should be clearly explained, and users should be made aware of its limitations, especially in high-stakes scenarios like policy-making or financial decisions.
Privacy: The model is trained on publicly available census data and does not involve personally identifiable information (PII).
## Caveats and Recommendations
Generalizability: The model is trained on historical census data and may not perform as well on populations with significant socio-economic changes.
Recommendations for Improvement:
Use more recent data to improve generalizability.
Explore models that are more interpretable for high-stakes decisions (e.g., decision trees or linear models).
Evaluate the model further for fairness across different demographic groups to ensure that it is not perpetuating or exacerbating inequalities.