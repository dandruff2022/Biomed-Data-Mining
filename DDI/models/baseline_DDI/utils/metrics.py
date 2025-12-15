import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_and_save(model, texts, X_test, y_test, out_dir):
    """
    Evaluate model and save:
        - accuracy
        - macro-F1
        - weighted-F1
        - classification report
        - raw confusion matrix
        - normalized confusion matrix heatmap

    Returns:
        test_acc, macro_f1, cm_array
    """

    os.makedirs(out_dir, exist_ok=True)

    # Handle both SVMModel (takes only X) and LogisticModel (takes texts and X)
    try:
        preds = model.predict(texts, X_test)
    except TypeError:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")

    report_str = classification_report(
        y_test,
        preds,
        digits=4
    )
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[INFO] Saved classification_report.txt")

    cm = confusion_matrix(y_test, preds)
    labels = np.unique(y_test)

    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix (Raw Counts)")
    plt.axis('off')

    table = plt.table(
        cellText=cm,
        rowLabels=labels,
        colLabels=labels,
        loc='center',
        cellLoc='center'
    )
    table.scale(1, 2)

    plt.savefig(os.path.join(out_dir, "confusion_matrix_raw.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion_matrix_raw.png")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,   
        vmax=1    
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion_matrix_normalized.png")

    # save raw cm array
    np.save(os.path.join(out_dir, "confusion_matrix_raw.npy"), cm)

    return acc, macro_f1, cm
