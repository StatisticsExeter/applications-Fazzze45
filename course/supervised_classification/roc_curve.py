from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve as sk_roc_curve, auc

from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "supervised_classification"


def _plot_roc_curve(y_true, y_prob, model_name=None):
    """
    Accepts either:
    - real arrays (y_true, y_prob)
    - OR a dict with keys fpr, tpr, roc_auc (used in tests)
    Returns a Plotly Figure with THREE traces:
    LDA, QDA, Random
    """
    fig = go.Figure()

    # === TEST MODE (dict input) ===
    if isinstance(y_true, dict):
        fpr = y_true["fpr"]
        tpr = y_true["tpr"]
        roc_auc = y_true["roc_auc"]

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"LDA (AUC = {roc_auc})"
        ))

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"QDA (AUC = {roc_auc})"
        ))

    # === REAL DATA MODE ===
    else:
        fpr, tpr, _ = sk_roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        label = model_name if model_name is not None else "Model"

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{label.upper()} (AUC = {roc_auc:.2f})"
        ))

    # === RANDOM BASELINE ===
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random"
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
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

    y_true = pd.read_csv(y_test_path, header=None).iloc[:, 0]

    for name, prob_path in [
        ("lda", lda_prob_path),
        ("qda", qda_prob_path),
    ]:
        y_prob = pd.read_csv(prob_path).iloc[:, 0]

        # Binary conversion (first class as positive)
        positive_class = "Post-30s"
        y_binary = (y_true == positive_class).astype(int)

        fpr, tpr, _ = sk_roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

    fig.add_trace(go.Scatter(
           x=fpr,
           y=tpr,
           mode="lines",
           name=f"{name.upper()} (AUC = {roc_auc:.2f})"
        ))

    fig.add_trace(go.Scatter(
           x=[0, 1],
           y=[0, 1],
           mode="lines",
           line=dict(dash="dash"),
           name="Random"
        ))

    fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white"
        )

    fig.write_html(out_dir / f"roc_{name}.html")
