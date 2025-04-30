# P2P Lending Data Analysis and Grade Prediction (LendingClub)

This repository contains Python code for analyzing LendingClub loan data from 2017, focusing on exploring default rates and building predictive models for loan grades.
This project was completed as part of an assignment exploring the relationship between Peer-to-Peer (P2P) lending and traditional banking.

## Project Overview

The analysis uses the LendingClub `loan_data_2017.csv` dataset to:
1.  Visualize the relationship between loan grade and default rate.
2.  Demonstrate data splitting techniques for model training and testing.
3.  Build and evaluate machine learning models (Logistic Regression and Random Forest) to predict whether a loan applicant will receive a "good" grade (A or B) versus a "bad" grade (C-G).
4.  Identify key features influencing loan grade prediction, with careful consideration to avoid data leakage.

## Features / Tasks Completed

*   **Task 1: Default Rate Analysis**
    *   Calculates the default rate for each loan grade category (A through G).
    *   Generates a bar plot visualizing these default rates (`/task1_default_rates.png`).
    *   Uses standard Python libraries (`csv`, `matplotlib`, `collections`).
*   **Task 2: Data Splitting**
    *   Loads the dataset using `pandas`.
    *   Uses the `pandas.sample()` method to randomly split the data into a 70% training set and a 30% testing set, as per assignment instructions.
    *   Saves the split datasets (`results/train_data.csv`, `results/test_data.csv`).
*   **Task 3: Loan Grade Prediction**
    *   Defines a binary classification task: Good Grade (A, B) vs. Bad Grade (C, G).
    *   **Feature Engineering & Selection:** Selects features available *before* loan origination to prevent data leakage (e.g., excluding `interest_rate`, `loan_status`). Transforms features like `earliest_cr_line` and cleans others.
    *   **Modeling:**
        *   Builds a Logistic Regression model (interpretable baseline).
        *   Builds a Random Forest Classifier model (more complex model, targeting bonus points).
    *   **Preprocessing:** Uses `scikit-learn` Pipelines for robust imputation (median) and scaling (StandardScaler).
    *   **Evaluation:** Evaluates models using ROC AUC, Classification Reports (Precision, Recall, F1-score), Cross-Validation, and Confusion Matrices. Handles class imbalance using `class_weight='balanced'`.
    *   **Interpretation:** Extracts and analyzes feature importances from the Random Forest model to identify key predictors.
    *   **Outputs:** Generates evaluation plots (ROC, Precision-Recall, Confusion Matrices) and feature importance analysis saved in the `results/` directory.
