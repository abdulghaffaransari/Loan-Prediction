# Loan Prediction Model

This project involves developing and evaluating machine learning models to predict loan approval status using a dataset of loan applicants. The models include Random Forest Classifier, Logistic Regression, and Decision Tree Classifier. The project utilizes MLflow for tracking experiments and model performance.

## Project Overview

The goal of this project is to predict whether a loan application will be approved or not based on various features of the applicants. The dataset used in this project contains various features related to the applicants' financial information and loan status.

## Dataset

- **File**: `train.csv`
- **Description**: Contains information about loan applicants, including both numerical and categorical features.
- **Features**:
  - `Loan_ID`: Unique identifier for each loan application (not used in modeling).
  - `Loan_Status`: Target variable indicating whether the loan was approved or not (1 for approved, 0 for not approved).
  - `ApplicantIncome`: Monthly income of the applicant.
  - `CoapplicantIncome`: Monthly income of the co-applicant.
  - `LoanAmount`: The loan amount applied for.
  - Other categorical and numerical features related to the applicant's details.

## Installation

To run this project, you'll need the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- mlflow

You can install these packages using pip:

## Usage

### Data Preprocessing

1. **Load the dataset**: Handle missing values by filling categorical columns with mode and numerical columns with median.
2. **Handle Outliers**: Cap extreme values in numerical columns to manage outliers.
3. **Log Transformation**: Apply log transformation to the `LoanAmount` and `TotalIncome` features.
4. **Feature Selection**: Drop the `ApplicantIncome` and `CoapplicantIncome` columns.
5. **Encoding**: Encode categorical variables and the target variable (`Loan_Status`).

### Model Training

1. **Train-Test Split**: Divide the dataset into training and testing sets.
2. **Model Training**: Train and tune three classifiers:
   - Random Forest
   - Logistic Regression
   - Decision Tree
   Using GridSearchCV for hyperparameter tuning.

### Model Evaluation

1. **Evaluation Metrics**: Assess models using accuracy, F1-score, and AUC-ROC curve.
2. **Logging**: Save the ROC curve plot and log metrics and model parameters using MLflow.

### MLflow Tracking

- **Tracking**: MLflow is used to track experiments, log model parameters, metrics, and artifacts.

