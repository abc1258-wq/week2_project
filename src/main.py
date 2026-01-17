from data_ingestion import load_dataset
from data_quality import check_data_quality
from etl_pipeline import run_etl
from eda_visualization import plot_eda

print("WEEK 2 – DATA ANALYTICS PIPELINE")

df = load_dataset("data/KDDTrain.txt")
df = check_data_quality(df)
data_transformed = run_etl(df)
plot_eda(df, data_transformed)

print("\nWEEK 2 COMPLETED SUCCESSFULLY")
