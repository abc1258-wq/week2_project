import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    print(f"\n{model_name} Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    if model_name == "Random Forest":
        disp.plot(cmap="Greens")
    else:
        disp.plot(cmap="Blues")

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return metrics


def compare_models(lr_metrics, rf_metrics):

    models = ["Logistic Regression", "Random Forest"]
    f1_scores = [
        lr_metrics["F1-score"],
        rf_metrics["F1-score"]
    ]
    accuracies = [
        lr_metrics["Accuracy"],
        rf_metrics["Accuracy"]
    ]

    x = range(len(models))

    plt.figure(figsize=(7,5))

    # Bars for F1-score
    plt.bar(
        x,
        f1_scores,
        color=["#FF7F0E", "#1F77B4"],
        alpha=0.85,
        label="F1-score"
    )

    # Points for Accuracy
    plt.scatter(
        x,
        accuracies,
        color="black",
        s=80,
        zorder=3,
        label="Accuracy"
    )

    # Add accuracy value labels (4 decimal places, exact)
    for i, acc in enumerate(accuracies):
        plt.text(
            i,
            acc - 0.03,
            f"{acc:.4f}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(x, models)
    plt.ylabel("Score")

    # IMPORTANT: limit y-axis slightly above 1
    plt.ylim(0, 1.01)

    plt.title("Model Comparison: F1-score (bars) vs Accuracy (points)")
    plt.legend()
    plt.tight_layout()
    plt.show()

