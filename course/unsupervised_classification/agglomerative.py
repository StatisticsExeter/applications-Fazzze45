from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from course.utils import find_project_root


VIGNETTE_DIR = (
    Path("data_cache")
    / "vignettes"
    / "unsupervised_classification"
)


def agglomerative_clustering(n_clusters: int = 3):
    """Fit complete-linkage agglomerative clustering and save a PCA plot."""

    base_dir = find_project_root()

    data_path = base_dir / "data_cache" / "la_collision.csv"
    output_dir = base_dir / VIGNETTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric variables were found for clustering.")

    n_clusters = min(n_clusters, len(numeric_df))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="complete",
    )

    labels = model.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    plot_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"],
    )
    plot_df["Cluster"] = labels.astype(str)

    if "lad_ons" in df.columns:
        plot_df["Local Authority"] = df["lad_ons"].astype(str)

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_name=(
            "Local Authority"
            if "Local Authority" in plot_df.columns
            else None
        ),
        title="Agglomerative Clustering by PCA Components",
    )

    output_path = output_dir / "agglomerative_scatter.html"
    fig.write_html(output_path)

    results = pd.DataFrame(
        {
            "lad_ons": (
                df["lad_ons"]
                if "lad_ons" in df.columns
                else range(len(df))
            ),
            "agglomerative_cluster": labels,
        }
    )

    results.to_csv(
        output_dir / "agglomerative_clusters.csv",
        index=False,
    )

    print(f"Agglomerative plot saved to: {output_path}")
    return None
