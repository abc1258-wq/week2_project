def check_data_quality(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    df["attack"] = df["label"].apply(lambda x: "Normal" if x == "normal" else "Attack")
    print("\nAttack label distribution:")
    print(df["attack"].value_counts())

    return df
