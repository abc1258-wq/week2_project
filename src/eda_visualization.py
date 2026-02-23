import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

def plot_eda(df, data_transformed):

    os.makedirs("results", exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("talk")

    numeric_df = data_transformed.select_dtypes(include=["int64", "float64"])

    # Remove constant columns (important for clustering)
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    print("Generating Ultra Advanced Visualizations...")

    #  PCA 2D Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_df.sample(5000))

    plt.figure(figsize=(8,6))
    plt.scatter(pca_result[:,0],
                pca_result[:,1],
                alpha=0.6,
                c="darkcyan")
    plt.title("PCA 2D Projection")
    plt.tight_layout()
    plt.savefig("results/pca_plot.png")
    plt.close()

    #  Rolling Mean Trend
    rolling = numeric_df.iloc[:,0].rolling(window=50).mean()
    plt.figure(figsize=(8,6))
    plt.plot(rolling, color="green")
    plt.title("Rolling Mean Trend")
    plt.tight_layout()
    plt.savefig("results/rolling_mean.png")
    plt.close()

    # Outlier Highlight Plot
    col = numeric_df.iloc[:,0]
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1

    outliers = col[(col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr)]

    plt.figure(figsize=(8,6))
    plt.scatter(col.index, col, alpha=0.3)
    plt.scatter(outliers.index, outliers, color="red")
    plt.title("Outlier Highlight Plot")
    plt.tight_layout()
    plt.savefig("results/outlier_plot.png")
    plt.close()

    # Skewness Plot
    skew_vals = numeric_df.skew().sort_values(ascending=False)[:15]

    plt.figure(figsize=(10,6))
    sns.barplot(
        x=skew_vals.values,
        y=skew_vals.index,
        hue=skew_vals.index,
        legend=False,
        palette="mako"
    )
    plt.title("Top Skewed Features")
    plt.tight_layout()
    plt.savefig("results/skewness_plot.png")
    plt.close()

    #  Safe Correlation Matrix
    corr_matrix = numeric_df.corr()
    corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan)
    corr_matrix = corr_matrix.fillna(0)

    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix,
                cmap="vlag",
                center=0)
    plt.title("Correlation Heatmap (Cleaned)")
    plt.tight_layout()
    plt.savefig("results/correlation_heatmap_clean.png")
    plt.close()

    def plot_model_comparison(results_df):
        os.makedirs("results", exist_ok=True)

        sns.set_style("whitegrid")
        sns.set_context("talk")

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

        # Grouped Bar Comparison
        melted = results_df.melt(id_vars="Model",
                                 value_vars=metrics,
                                 var_name="Metric",
                                 value_name="Score")

        plt.figure(figsize=(10, 6))
        sns.barplot(data=melted,
                    x="Metric",
                    y="Score",
                    hue="Model",
                    palette="Set2")
        plt.title("Model Performance Comparison")
        plt.tight_layout()
        plt.savefig("results/grouped_model_comparison.png")
        plt.close()
    print("Ultra Advanced Visualizations Generated Successfully.")