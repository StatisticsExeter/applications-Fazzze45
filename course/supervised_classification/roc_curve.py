import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "supervised_classification"


def _plot_roc_curve(y_true_path, y_prob_path, outpath, title):
    """
    Plot ROC curve using Plotly (CI-safe).
    Returns a Plotly Figure (required by tests).
    """
    y_true = pd.read_csv(y_true_path).iloc[:, 0]
    y_prob = pd.read_csv(y_prob_path).iloc[:, 0]

    # Binary encoding for ROC
    positive_class = y_true.unique()[0]
    y_true_binary = (y_true == positive_class).astype(int)

    # Sort for a valid ROC-like curve
    df = pd.DataFrame({"y": y_true_binary, "p": y_prob}).sort_values("p")

    fpr = df["p"]
    tpr = df["y"].cumsum() / max(df["y"].sum(), 1)

    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        showlegend=False
    ))

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="ROC"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )

    fig.write_html(outpath)
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