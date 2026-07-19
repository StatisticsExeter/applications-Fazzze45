from pathlib import Path

import pandas as pd
import plotly.express as px


def diagnostic_plots():
    base_dir = Path(".")
    input_path = base_dir / "data_cache" / "models" / "diagnostics.csv"
    output_dir = base_dir / "data_cache" / "vignettes" / "regression"

    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(input_path)

    residual_plot = px.scatter(
        diagnostics,
        x="fitted",
        y="residual",
        title="Residuals versus fitted values",
        labels={
            "fitted": "Fitted values",
            "residual": "Residuals",
        },
    )
    residual_plot.add_hline(y=0, line_dash="dash")
    residual_plot.write_html(output_dir / "residuals_vs_fitted.html")

    histogram = px.histogram(
        diagnostics,
        x="residual",
        nbins=30,
        title="Distribution of model residuals",
        labels={"residual": "Residual"},
    )
    histogram.write_html(output_dir / "residual_histogram.html")


if __name__ == "__main__":
    diagnostic_plots()
