import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# Model names
# ---------------------------------
models = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Random Forest (DS2)",
    "KNN",
    "Decision Tree"
]

# ---------------------------------
# Accuracy values (final)
# ---------------------------------
accuracy_scores = [0.895, 0.999, 0.905, 0.950, 0.939, 0.969]

x = np.arange(len(models))

# ---------------------------------
# Create figure
# ---------------------------------
plt.figure(figsize=(12, 5))

bars = plt.bar(
    x,
    accuracy_scores,
    color=[
        "#1f77b4",
        "#d9d9d9",
        "#ff7f0e",
        "#8c564b",
        "#2ca02c",
        "#d62728"
    ],
    width=0.6
)

# ---------------------------------
# Annotate accuracy values on bars
# ---------------------------------
for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# ---------------------------------
# Formatting
# ---------------------------------
plt.title("Accuracy Comparison Across 6 Models")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.02)
plt.xticks(x, models, rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
