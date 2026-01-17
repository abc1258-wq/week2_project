import pandas as pd

def run_etl(df):
    categorical_cols = ["protocol_type", "service", "flag"]
    data_transformed = pd.get_dummies(df, columns=categorical_cols)

    data_transformed.to_csv("week2_processed_data.csv", index=False)
    print("\nETL completed")
    print("Processed dataset saved as week2_processed_data.csv")
    print("New shape:", data_transformed.shape)

    return data_transformed
