import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sk_roc_curve, auc
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "supervised_classification"


def _plot_roc_curve(y_true_path, y_prob_path, outpath, title):
    y_true = pd.read_csv(y_true_path).iloc[:, 0]
    y_prob = pd.read_csv(y_prob_path).iloc[:, 0]

    # Convert multiclass labels to binary
    positive_class = y_true.unique()[0]
    y_true_binary = (y_true == positive_class).astype(int)

    fpr, tpr, _ = sk_roc_curve(y_true_binary, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (Positive class: {positive_class})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def roc_curve():
    """
    Entry point for doit — MUST take no arguments
    """
    base_dir = find_project_root()

    y_test = base_dir / "data_cache" / "energy_y_test.csv"

    lda_prob = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    qda_prob = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"

    out_lda = base_dir / VIGNETTE_DIR / "roc_lda.png"
    out_qda = base_dir / VIGNETTE_DIR / "roc_qda.png"

    _plot_roc_curve(y_test, lda_prob, out_lda, "LDA ROC Curve")
    _plot_roc_curve(y_test, qda_prob, out_qda, "QDA ROC Curve")