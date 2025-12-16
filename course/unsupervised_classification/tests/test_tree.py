import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from course.unsupervised_classification.tree import (
    _plot_dendrogram, _cutree, _pca, _scatter_clusters)


def test_plot_dendrogram_returns_figure():
    df = pd.DataFrame(np.random.rand(10, 4))
    fig = _plot_dendrogram(df)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Interactive Hierarchical Clustering Dendrogram"


def test_cutree_returns_dataframe():
    df = pd.DataFrame(np.random.rand(10, 4))
    tree = linkage(df, method='ward')
    clusters_df = _cutree(tree, height=5)
    assert isinstance(clusters_df, pd.DataFrame)
    assert 'cluster' in clusters_df.columns
    assert clusters_df.shape[0] == df.shape[0]


def test_pca_returns_two_components():
    df = pd.DataFrame(np.random.rand(10, 5))
    df_pca = _pca(df)
    assert isinstance(df_pca, pd.DataFrame)
    assert df_pca.shape == (10, 2)
    assert list(df_pca.columns) == ['PC1', 'PC2']


def test_scatter_clusters_returns_figure():
    df = pd.DataFrame({
        'PC1': np.random.rand(10),
        'PC2': np.random.rand(10),
        'cluster': np.random.choice(['A', 'B'], size=10)
    })
    fig = _scatter_clusters(df)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'PCA Scatter Plot Colored by Cluster Labels'
