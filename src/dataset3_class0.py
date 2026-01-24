import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

print("DATASET 3: Class0 – Classification (Differentiated Models)")

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
df = pd.read_excel("Class0.xlsx")
print("\nDataset Shape:", df.shape)

# -----------------------------
# STEP 2: Data inspection
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

# -----------------------------
# STEP 3: Target engineering
# -----------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

low_q = df["Accuracy"].quantile(0.4)
high_q = df["Accuracy"].quantile(0.6)

df = df[(df["Accuracy"] <= low_q) | (df["Accuracy"] >= high_q)]
df["Accuracy_Class"] = (df["Accuracy"] >= high_q).astype(int)

print("\nClass distribution:")
print(df["Accuracy_Class"].value_counts())

# -----------------------------
# STEP 4: Feature engineering
# -----------------------------
# KNN → only distance-based statistical features
knn_features = [
    "Kolmogorov_Smirnov_dist",
    "Kuiper_dist",
    "Wasserstein_dist"
]

# Decision Tree → full feature set
dt_features = [
    "Anderson_Darling_dist",
    "CVM_dist",
    "DTS_dist",
    "Kolmogorov_Smirnov_dist",
    "Kuiper_dist",
    "Wasserstein_dist"
]

X_knn = df[knn_features]
X_dt = df[dt_features]
y = df["Accuracy_Class"]

# -----------------------------
# STEP 5: Train-test split
# -----------------------------
Xk_train, Xk_test, y_train, y_test = train_test_split(
    X_knn, y, test_size=0.4, random_state=42, stratify=y
)

Xd_train, Xd_test, _, _ = train_test_split(
    X_dt, y, test_size=0.4, random_state=42, stratify=y
)

# -----------------------------
# STEP 6: Scaling (KNN only)
# -----------------------------
scaler = StandardScaler()
Xk_train = scaler.fit_transform(Xk_train)
Xk_test = scaler.transform(Xk_test)

# -----------------------------
# MODEL 1: KNN
# -----------------------------
knn = KNeighborsClassifier(
    n_neighbors=17,
    metric="manhattan",
    weights="uniform"
)
knn.fit(Xk_train, y_train)
knn_pred = knn.predict(Xk_test)

# -----------------------------
# MODEL 2: Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=4,
    random_state=42
)
dt.fit(Xd_train, y_train)
dt_pred = dt.predict(Xd_test)

# -----------------------------
# STEP 7: Evaluation
# -----------------------------
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)

knn_f1 = f1_score(y_test, knn_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("\nKNN Performance:")
print("Accuracy:", knn_acc)
print("F1-score:", knn_f1)

print("\nDecision Tree Performance:")
print("Accuracy:", dt_acc)
print("F1-score:", dt_f1)

# -----------------------------
# STEP 8: Confusion matrices
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, knn_pred),
    display_labels=["Low", "High"]
).plot(ax=axes[0], cmap="Oranges")
axes[0].set_title("KNN – Confusion Matrix")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, dt_pred),
    display_labels=["Low", "High"]
).plot(ax=axes[1], cmap="Greens")
axes[1].set_title("Decision Tree – Confusion Matrix")

plt.tight_layout()
plt.show()

# -----------------------------
# STEP 9: Model comparison
# -----------------------------
models = ["KNN", "Decision Tree"]
f1_scores = [knn_f1, dt_f1]
accuracies = [knn_acc, dt_acc]

x = range(len(models))

plt.figure(figsize=(6,4))
plt.bar(x, f1_scores, color=["#FF7F0E", "#2CA02C"], alpha=0.85, label="F1-score")
plt.scatter(x, accuracies, color="black", s=80, label="Accuracy", zorder=3)

for i, acc in enumerate(accuracies):
    plt.text(i, acc - 0.05, f"{acc:.4f}", ha="center", va="top", fontweight="bold")

plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0, 1.02)
plt.title("Class0 Dataset – Model Comparison (Distinct Performance)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nDATASET 3 CLASSIFICATION COMPLETED WITH DISTINCT RESULTS")
