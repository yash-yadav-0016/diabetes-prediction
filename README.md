# Diabetes Risk Classification Model

## Objective
Develop a classification model to predict diabetes risk using medical patient data.

## Dataset
- **Source:** Medical Patient Dataset
- **Records:** 768 patients
- **Features:** 8 health metrics (glucose, blood pressure, BMI, age, etc.)
- **Target:** Diabetes diagnosis (binary classification)

## Methodology
1. Data preprocessing: Feature scaling (StandardScaler), missing value handling
2. Exploratory data analysis with statistical summaries
3. Model comparison: Logistic Regression vs Random Forest
4. Evaluation: 5-fold cross-validation, accuracy metrics

## Results
- **Logistic Regression Accuracy:** 75%
- **Random Forest Accuracy:** 82%
- **Cross-validation Score:** 81% ± 2%
- **Improvement:** +7% over Logistic Regression

## Technologies
- Python 3.x
- Scikit-learn (ML algorithms)
- Pandas (data handling)

## Key Findings
- Random Forest captures non-linear patterns better than Logistic Regression
- Top predictive features: Glucose, BMI, Age
- Model achieves clinically relevant accuracy for risk screening
