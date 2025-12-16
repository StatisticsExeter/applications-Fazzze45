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
    """Plot cluster centers in two separate grouped bar plots."""
    original_centers = scaler.inverse_transform(scaled_centers)

    # First half of columns
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, [0]]
    centers_df['cluster'] = [f'Cluster {i}' for i in range(k)]
    centers_melted = centers_df.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig1 = px.bar(
        centers_melted,
        x='Feature',
        y='Value',
        color='cluster',
        barmode='group',
        title='Cluster Centers by Feature (Original Scale)'
    )

    # Second half of columns
    centers_df = pd.DataFrame(original_centers, columns=colnames).iloc[:, 1:]
    centers_df['cluster'] = [f'Cluster {i}' for i in range(k)]
    centers_melted = centers_df.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig2 = px.bar(
        centers_melted,
        x='Feature',
        y='Value',
        color='cluster',
        barmode='group',
        title='Cluster Centers by Feature (Original Scale)'
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