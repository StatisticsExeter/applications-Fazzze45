
from pathlib import Path
import pandas as pd
import plotly.express as px
from course.utils import find_project_root

# Path to vignette directory
VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def _scatter(df, title):
    """When called with dataframe `df` and a string `title`,
    return a plotly express object which is a scatterplot of all numeric variables in the dataframe.
    The title should be as provided in the function call.
    """
    numeric_df = df.select_dtypes(include="number").copy()
    fig = px.scatter_matrix(
        numeric_df,
        dimensions=numeric_df.columns,
        title=title,
        color_continuous_scale="Viridis",
        height=900,
        width=900
    )
    return fig


def plot_scatter():
    """Load data and create scatterplot"""
    base_dir = find_project_root()
    csv_path = base_dir / 'data_cache' / 'la_collision.csv'
    print("📂 Loading data from:", csv_path)

    df = pd.read_csv(csv_path, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
    print("✅ Data loaded successfully with shape:", df.shape)

    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    print("💾 Saving scatterplot to:", outpath)

    fig = _scatter(df, "Crash types in each Local Authority")
    fig.write_html(outpath)
    print("🎉 Scatterplot successfully written to file!")


if __name__ == "__main__":
    plot_scatter()
