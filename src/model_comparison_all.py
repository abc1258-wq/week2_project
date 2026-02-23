import os
import matplotlib.pyplot as plt

def plot_model_comparison(results_df):

    os.makedirs("results", exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for metric in metrics:
        plt.figure()
        plt.bar(results_df["Model"], results_df[metric])
        plt.title(f"{metric} Comparison")
        plt.savefig(f"results/{metric}_comparison.png")
        plt.close()