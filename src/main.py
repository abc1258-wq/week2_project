import pandas as pd
import os
from data_ingestion import load_dataset
from data_quality import check_data_quality
from etl_pipeline import run_etl
from model_training import train_models
from model_evaluation import evaluate_models
from model_comparison_all import plot_model_comparison
from eda_visualization import plot_eda

print("WEEK 2 - DATA ANALYTICS PIPELINE")

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = load_dataset("data/KDDTrain.txt")

# -------------------------------
# 2. Data Quality Check
# -------------------------------
df = check_data_quality(df)

# -------------------------------
# 3. ETL Processing
# -------------------------------
data_transformed = run_etl(df)

# -------------------------------
# 4. Generate EDA Graphs
# -------------------------------
plot_eda(df, data_transformed)

# -------------------------------
# 5. Separate Features & Target
# -------------------------------
X = data_transformed.drop("attack", axis=1)
y = data_transformed["attack"]

# Convert categorical columns to numeric
X = pd.get_dummies(X)

# Convert target to numeric (0/1)
y = y.map({"Normal": 0, "Attack": 1})

# -------------------------------
# 6. Train Model
# -------------------------------
models, X_test, y_test = train_models(X, y)

# -------------------------------
# 7. Evaluate Model & Generate Graphs
# -------------------------------
results = evaluate_models(models, X_test, y_test)

# -------------------------------
# 8. Model Comparison Graph
# -------------------------------
plot_model_comparison(results)

print(results)

print("COMPLETED SUCCESSFULLY")