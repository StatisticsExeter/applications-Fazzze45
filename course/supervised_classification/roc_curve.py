import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc


def _plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.2f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance"))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve",
    )
    return fig


def roc_curve():
    """
    doit entry point — MUST take no arguments
    """
    base_dir = find_project_root()

    y_test = base_dir / "data_cache" / "energy_y_test.csv"

    lda_prob = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    qda_prob = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"

    out_lda = base_dir / VIGNETTE_DIR / "roc_lda.html"
    out_qda = base_dir / VIGNETTE_DIR / "roc_qda.html"

    _plot_roc_curve(y_test, lda_prob, out_lda, "LDA ROC Curve")
    _plot_roc_curve(y_test, qda_prob, out_qda, "QDA ROC Curve")
