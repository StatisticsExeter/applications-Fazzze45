from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve as sk_roc_curve, auc

from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "supervised_classification"


def _plot_roc_curve(y_true, y_prob):
    """
    Used by tests.

    y_true: dict with keys ['fpr', 'tpr', 'roc_auc']
    y_prob: unused but required by test signature
    """
    fig = go.Figure()

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=y_true["fpr"],
            y=y_true["tpr"],
            mode="lines",
            name="ROC Curve",
        )
    )

    # Chance line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash"),
        )
    )

    # AUC label (empty trace, just for legend)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"AUC = {y_true['roc_auc']}",
        )
    )

    return fig


def roc_curve():
    """
    doit entry point — MUST take no arguments
    """
    base_dir = find_project_root()

    y_test_path = base_dir / "data_cache" / "energy_y_test.csv"
    lda_prob_path = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    qda_prob_path = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"

    out_dir = base_dir / VIGNETTE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.read_csv(y_test_path).iloc[:, 0]

    for name, prob_path in [
        ("lda", lda_prob_path),
        ("qda", qda_prob_path),
    ]:
        y_prob = pd.read_csv(prob_path).iloc[:, 0]

        # Binary conversion (first class as positive)
        positive_class = y_true.unique()[0]
        y_binary = (y_true == positive_class).astype(int)

        fpr, tpr, _ = sk_roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)

        fig = _plot_roc_curve(
            {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc},
            None,
        )

        fig.write_html(out_dir / f"roc_{name}.html")
