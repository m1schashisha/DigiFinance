# Task 3: Loan Grade Binary Classification
# Objective: Predict whether a loan will be classified as "good grade" (A or B) or "bad grade" (C-G)
# Features are selected based on their relevance to credit risk assessment
# Multiple ML models are compared to find the best performing approach

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_loan_data(df):
    """
    Prepare loan data for grade prediction
    
    Process:
    1. Create binary target variable (good_grades: A&B=1, others=0)
    2. Clean and preprocess features
    3. Handle categorical variables
    4. Create derived features (e.g., credit history length)
    
    Selected Features Rationale:
    - annual_inc: Direct indicator of repayment capability
    - dti: Measures existing financial burden
    - emp_length: Shows job stability
    - loan_amnt: Indicates risk exposure level
    - int_rate: Direct risk assessment metric
    - verification_status: Data reliability indicator
    - total_acc: Credit experience measure
    - earliest_cr_line: Credit history length
    - home_ownership: Asset ownership status
    - purpose: Loan intention impact
    """
    # Create binary classification target
    df['good_grades'] = df['grade'].apply(lambda x: 1 if x in ['A', 'B'] else 0)

    # Clean numerical features
    df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)

    # Select and document relevant features
    features = [
        'annual_inc',          # Higher income → better repayment capability
        'dti',                 # Lower DTI → better financial health
        'emp_length',          # Longer employment → more stability
        'loan_amnt',          # Loan size affects risk level
        'int_rate',           # Interest rate reflects assessed risk
        'verification_status', # Verified info → more reliable assessment
        'total_acc',          # More accounts → more credit experience
        'earliest_cr_line',   # Longer credit history → more reliable
        'home_ownership',     # Property ownership → financial stability
        'purpose'             # Loan purpose → risk variation
    ]

    # Handle categorical variables with one-hot encoding
    categorical_features = ['verification_status', 'home_ownership', 'purpose']
    df_encoded = pd.get_dummies(df[features], columns=categorical_features)

    # Create credit history length feature
    current_date = pd.Timestamp.now()
    df_encoded['credit_history_length'] = df_encoded['earliest_cr_line'].apply(
        lambda x: (current_date - pd.to_datetime(x, format='%b-%Y')).days / 365.25
    )
    df_encoded.drop('earliest_cr_line', axis=1, inplace=True)
    
    # Convert employment length to numeric
    df_encoded['emp_length'] = df_encoded['emp_length'].str.extract(r'(\d+)').astype(float)
    
    return df_encoded, df['good_grades']

def create_models(dataset_size=None):
    """
    Creates multiple ML pipelines for model comparison
    
    Models Selected:
    1. Random Forest: Handles non-linear relationships, feature importance
    2. Gradient Boosting: Strong predictive power, handles imbalanced data
    3. Logistic Regression: Simple, interpretable baseline model
    4. SVM: Good for complex decision boundaries (optional for large datasets)
    
    Args:
        dataset_size: Number of samples in dataset, used to determine if SVM should be included
    
    Each pipeline includes:
    - Imputer for missing values
    - StandardScaler for feature scaling
    - Model with optimized parameters
    """
    # Define preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())                    # Normalize features
    ])
    
    # Create dictionary of model pipelines
    models = {
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=100,     # Number of trees
                random_state=42,      # For reproducibility
                class_weight='balanced',  # Handle class imbalance
                n_jobs=-1            # Use all CPU cores
            ))
        ]),
        'gradient_boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                subsample=0.8        # Prevent overfitting
            ))
        ]),
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(
                max_iter=1000,        # Increased iterations for convergence
                random_state=42,
                class_weight='balanced'
            ))
        ])
    }
    
    # Only add SVM for smaller datasets (optional)
    if dataset_size is not None and dataset_size < 10000:  # Adjust threshold based on your system's capabilities
        models['svm'] = Pipeline([
            ('preprocessor', preprocessor),
            ('model', SVC(
                kernel='rbf',
                random_state=42,
                probability=True,
                class_weight='balanced',
                max_iter=1000
            ))
        ])
    
    return models

# Model evaluation and feature importance visualization
def evaluate_models(models, X_train, y_train, X_test, y_test, feature_names):
    """
    Evaluate multiple models and compare their performance
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'classification_report': class_report,
            'cv_scores': cv_scores,
            'feature_importance': None
        }
        
        # Get feature importance if available
        if hasattr(model[-1], 'feature_importances_'):
            results[name]['feature_importance'] = model[-1].feature_importances_
            
            # Create feature importance plot for this model
            plt.figure(figsize=(12,6))
            plt.title(f"Feature Importance - {name}")
            importances = model[-1].feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.bar(range(X_train.shape[1]), importances[indices])
            plt.xticks(range(X_train.shape[1]), 
                      [feature_names[i] for i in indices], 
                      rotation=45, 
                      ha='right')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name}.png')
            plt.close()
    
    return results

def save_comparison_results(results, feature_names):
    """
    Save comparison results of multiple models to a file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'model_comparison_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("=== Loan Grade Prediction Model Comparison ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Compare accuracies
        f.write("=== Model Accuracies ===\n")
        for name, result in results.items():
            f.write(f"{name}: {result['accuracy']:.4f}\n")
        f.write("\n")
        
        # Detailed results for each model
        for name, result in results.items():
            f.write(f"=== Detailed Results for {name} ===\n")
            f.write("Classification Report:\n")
            f.write(result['classification_report'])
            f.write("\n")
            
            f.write("Cross-validation Scores:\n")
            cv_scores = result['cv_scores']
            f.write(f"Scores: {cv_scores}\n")
            f.write(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
            
            if result['feature_importance'] is not None:
                f.write("Top 10 Most Important Features:\n")
                indices = np.argsort(result['feature_importance'])[::-1][:10]
                for idx in indices:
                    f.write(f"{feature_names[idx]}: {result['feature_importance'][idx]:.4f}\n")
                f.write("\n")
            
        # Overall comparison summary
        f.write("=== Model Comparison Summary ===\n")
        accuracies = {name: result['accuracy'] for name, result in results.items()}
        cv_means = {name: result['cv_scores'].mean() for name, result in results.items()}
        
        best_accuracy = max(accuracies.items(), key=lambda x: x[1])
        best_cv = max(cv_means.items(), key=lambda x: x[1])
        
        f.write(f"\nBest Test Accuracy: {best_accuracy[0]} ({best_accuracy[1]:.4f})")
        f.write(f"\nBest CV Score: {best_cv[0]} ({best_cv[1]:.4f})")

def main():
    """
    Main execution pipeline:
    1. Load and prepare data
    2. Split into train/test sets
    3. Train and evaluate multiple models
    4. Compare model performance
    5. Save detailed results and visualizations
    """
    print("Loading loan data...")
    df = pd.read_csv('data/loan_data_2017.csv', low_memory=False)
    
    print("Preparing features...")
    X, y = prepare_loan_data(df)
    
    # Determine whether to include SVM based on dataset size
    dataset_size = len(X)
    print(f"Dataset size: {dataset_size} samples")
    if dataset_size >= 10000:
        print("Note: SVM model excluded due to large dataset size")
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print("Training and evaluating models...")
    models = create_models(dataset_size)  # Pass dataset size to create_models
    results = evaluate_models(models, X_train, y_train, X_test, y_test, X.columns)
    
    print("\nSaving comparison results...")
    save_comparison_results(results, X.columns)
    print("Done! Check the model_comparison_*.txt file for detailed results.")

if __name__ == "__main__":
    main()