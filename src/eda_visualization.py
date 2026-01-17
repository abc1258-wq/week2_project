import matplotlib.pyplot as plt
import seaborn as sns

def plot_eda(df, data_transformed):

    # Count plot
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x="attack", data=df, palette=["#4CAF50", "#F44336"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom")
    plt.title("Distribution of Normal vs Attack Traffic")
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(7,4))
    sns.boxplot(x="attack", y="src_bytes", data=df,
                palette=["#4CAF50", "#F44336"], showfliers=False)
    plt.title("Source Bytes Distribution by Traffic Type")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    numeric_data = data_transformed.select_dtypes(include="number")
    corr_matrix = numeric_data.corr()

    selected_features = [
        "duration","src_bytes","dst_bytes","count","srv_count",
        "serror_rate","srv_serror_rate","rerror_rate",
        "srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "dst_host_count","dst_host_srv_count"
    ]

    selected_features = [f for f in selected_features if f in corr_matrix.columns]

    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix.loc[selected_features, selected_features],
                cmap="RdBu_r", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Selected Network Traffic Features")
    plt.tight_layout()
    plt.show()

    # Pie chart
    attack_counts = df["attack"].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(attack_counts, labels=attack_counts.index,
            autopct="%1.1f%%", startangle=90,
            colors=["#1F77B4", "#FF7F0E"], explode=(0.03,0.03))
    plt.title("Proportion of Normal vs Attack Traffic")
    plt.show()
