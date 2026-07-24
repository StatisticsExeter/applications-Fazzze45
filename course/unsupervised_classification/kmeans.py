import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from course.utils import find_project_root
from course.unsupervised_classification.tree import _scatter_clusters, _pca

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def _kmeans(df, k):
    """Given dataframe df containing only suitable variables and integer k
    Return a scikit-learn KMeans solution fitted to these data."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    return kmeans


def _plot_centroids(scaled_centers, scaler, colnames, k):
    """Plot all cluster centres on scaled and original scales."""

    cluster_names = [f"Cluster {i + 1}" for i in range(k)]

    scaled_df = pd.DataFrame(
        scaled_centers,
        columns=colnames,
    )
    scaled_df["cluster"] = cluster_names

    scaled_long = scaled_df.melt(
        id_vars="cluster",
        var_name="Feature",
        value_name="Value",
    )

    fig1 = px.bar(
        scaled_long,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title="Cluster Centers by Feature (Scaled Data)",
    )

    original_centers = scaler.inverse_transform(scaled_centers)

    original_df = pd.DataFrame(
        original_centers,
        columns=colnames,
    )
    original_df["cluster"] = cluster_names

    original_long = original_df.melt(
        id_vars="cluster",
        var_name="Feature",
        value_name="Value",
    )

    fig2 = px.bar(
        original_long,
        x="Feature",
        y="Value",
        color="cluster",
        barmode="group",
        title="Cluster Centers by Feature (Original Scale)",
    )

    return fig1, fig2


def kmeans(k):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    # ✅ IMPORTANT: keep only numeric columns
    df = df.select_dtypes(include='number')

    # ✅ IMPORTANT: cap k so it never exceeds number of samples
    k = min(k, len(df))

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = _kmeans(df_scaled, k)
    clusters = kmeans.labels_
    scaled_centers = kmeans.cluster_centers_

    fig1, fig2 = _plot_centroids(scaled_centers, scaler, df.columns, k)

    outpath1 = base_dir / VIGNETTE_DIR / 'kcentroids1.html'
    outpath2 = base_dir / VIGNETTE_DIR / 'kcentroids2.html'
    fig1.write_html(outpath1)
    fig2.write_html(outpath2)

    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)

    outpath = base_dir / VIGNETTE_DIR / 'kscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)
