from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")

    return metrics, confusion_matrix(y_test, y_pred)


def save_metrics(all_metrics):
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nMetrikker lagret til: {out_dir / 'metrics.json'}")


def save_confusion_matrices(results):
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (_, cm) in results.items():
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=["On-time", "Late"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix – {name}")
        fig.savefig(out_dir / f"confusion_matrix_{name}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Confusion matrices lagret til: {out_dir}")


def run_evaluation(trained):
    all_metrics = {}
    results = {}

    for name, (model, scaler, X_test, y_test) in trained.items():
        metrics, cm = evaluate_model(name, model, X_test, y_test)
        all_metrics[name] = metrics
        results[name] = (metrics, cm)

    save_metrics(all_metrics)
    save_confusion_matrices(results)
    return all_metrics


if __name__ == "__main__":
    print("Kjør evaluate via main.py")