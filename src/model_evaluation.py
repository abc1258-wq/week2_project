import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_models(models, X_test, y_test):

    os.makedirs("results", exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("talk")

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            "Model": name,
            "Accuracy": report["accuracy"],
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1 Score": report["weighted avg"]["f1-score"]
        })

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/confusion_matrix.png")
        plt.close()

        # -----------------------------
        # ROC Curve
        # -----------------------------
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color="darkorange", linewidth=3)
        plt.plot([0,1],[0,1],'--', color="gray")
        plt.title(f"ROC Curve (AUC = {roc_auc:.4f})", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/roc_curve.png")
        plt.close()



        # -----------------------------
        # Feature Importance (FIXED)
        # -----------------------------
        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(20)

        plt.figure(figsize=(10,8))
        sns.barplot(
            data=feature_importance_df,
            x="Importance",
            y="Feature",
            hue="Feature",
            legend=False,
            palette="viridis"
        )
        plt.title("Top 20 Feature Importance", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        plt.close()

    return pd.DataFrame(results)