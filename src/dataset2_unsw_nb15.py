import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

print("DATASET 2: UNSW-NB15 – Attack Severity Classification")

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
df = pd.read_csv("UNSW-NB15_1.csv")
print("\nDataset Shape:", df.shape)

# -----------------------------
# STEP 2: Basic inspection
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum().head(15))

print("\nDescriptive statistics (numeric features):")
print(df.describe().head())

# -----------------------------
# STEP 3: Create binary attack severity label
# -----------------------------
# label == 1  -> Low severity attack (0)
# label > 1   -> High severity attack (1)

df["severity_label"] = (df["label"] > 1).astype(int)

print("\nBinary severity class distribution:")
print(df["severity_label"].value_counts())

# -----------------------------
# STEP 4: Feature / target split
# -----------------------------
y = df["severity_label"]

drop_cols = ["label", "severity_label"]
for col in ["id", "attack_cat"]:
    if col in df.columns:
        drop_cols.append(col)

X = df.drop(columns=drop_cols)

# Keep numeric features only
X = X.select_dtypes(include=["int64", "float64"])

print("\nFinal feature shape:", X.shape)

# -----------------------------
# STEP 5: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------
# STEP 6: Feature scaling (for SVM)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL 1: Support Vector Machine
# -----------------------------
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# -----------------------------
# MODEL 2: Random Forest (regularized)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# -----------------------------
# STEP 7: Evaluation
# -----------------------------
svm_acc = accuracy_score(y_test, svm_pred)
rf_acc = accuracy_score(y_test, rf_pred)

svm_f1 = f1_score(y_test, svm_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nSVM Performance:")
print("Accuracy:", svm_acc)
print("F1-score:", svm_f1)

print("\nRandom Forest Performance:")
print("Accuracy:", rf_acc)
print("F1-score:", rf_f1)

# -----------------------------
# STEP 8: Confusion matrices
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, svm_pred),
    display_labels=["Low Severity", "High Severity"]
).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("SVM – Confusion Matrix")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_pred),
    display_labels=["Low Severity", "High Severity"]
).plot(ax=axes[1], cmap="Greens")
axes[1].set_title("Random Forest – Confusion Matrix")

plt.tight_layout()
plt.show()

# -----------------------------
# STEP 9: Model comparison
# -----------------------------
models = ["SVM", "Random Forest"]
f1_scores = [svm_f1, rf_f1]
accuracies = [svm_acc, rf_acc]

x = range(len(models))

plt.figure(figsize=(6,4))
plt.bar(x, f1_scores, color=["#1F77B4", "#2CA02C"], alpha=0.85, label="F1-score")
plt.scatter(x, accuracies, color="black", s=80, label="Accuracy", zorder=3)

for i, acc in enumerate(accuracies):
    plt.text(i, acc - 0.03, f"{acc:.4f}", ha="center", va="top", fontweight="bold")

plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0, 1.02)
plt.title("UNSW-NB15 – Attack Severity Model Comparison")
plt.legend()
plt.tight_layout()
plt.show()

print("\nDATASET 2 (UNSW-NB15) SEVERITY CLASSIFICATION COMPLETED SUCCESSFULLY")
