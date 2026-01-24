import pandas as pd
from model_training import train_models
from model_evaluation import evaluate_model, compare_models

print("WEEK 3 – MODEL TRAINING AND COMPARISON")

# Load processed dataset from Week 2
data = pd.read_csv("week2_processed_data.csv")

# Train models
lr_model, rf_model, X_test, y_test = train_models(data)

# Evaluate Logistic Regression
lr_metrics = evaluate_model(
    lr_model, X_test, y_test, "Logistic Regression"
)

# Evaluate Random Forest
rf_metrics = evaluate_model(
    rf_model, X_test, y_test, "Random Forest"
)

# Compare models
compare_models(lr_metrics, rf_metrics)

print("\nWEEK 3 COMPLETED SUCCESSFULLY")
