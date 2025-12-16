from scipy.cluster.hierarchy import linkage, fcluster
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def fit_dendrogram(df):
    """
    Given a dataframe containing only suitable numeric values,
    return a scipy.cluster.hierarchy hierarchical clustering solution.
    """
    Z = linkage(df, method='ward')  # Ward’s linkage is default for this coursework
    return Z


def _plot_dendrogram(df):
    """Given a dataframe df containing only suitable variables,
    use plotly.figure_factory to plot a dendrogram of these data."""
    dend_fig = ff.create_dendrogram(df, orientation='top')
    dend_fig.update_layout(
        title="Interactive Hierarchical Clustering Dendrogram",
        height=600
    )
    return dend_fig


def _cutree(tree, height):
    """Given a scipy.cluster.hierarchy hierarchical clustering solution and a float of the height,
    cut the tree at that height and return the solution (cluster group membership) as a
    DataFrame with one column called 'cluster'."""
    clusters = fcluster(tree, height, criterion='distance')

    return pd.DataFrame({'cluster': clusters})


def _pca(df):
    """Given a dataframe of only suitable variables,
    return a dataframe of the first two PCA predictions (z values)
    with columns 'PC1' and 'PC2'."""
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df)
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])

    return df_pca


def _scatter_clusters(df):
    """Given a data frame containing columns 'PC1', 'PC2', and 'cluster'
    (the first two principal component projections and the cluster groups),
    return a Plotly Express scatterplot of PC1 versus PC2
    with marks to denote cluster group membership."""
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='cluster',
        title="PCA Scatter Plot Colored by Cluster Labels",
        labels={'cluster': 'Cluster'}
    )
    return fig


def hcluster_analysis():
    """
    End-to-end function to generate dendrogram and cluster scatter plots
    for the unsupervised learning task.
    """
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    # Standardize numeric features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include='number'))

    # Fit tree
    tree = fit_dendrogram(df_scaled)

    # Plot dendrogram
    fig_dendro = _plot_dendrogram(df_scaled)
    outpath_dendro = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig_dendro.write_html(outpath_dendro)

    # Choose a sensible height cut (you can tweak 6.0 depending on your data)
    height = 6.0
    df_clusters = _cutree(tree, height)

    # PCA for visualization
    df_pca = _pca(df.select_dtypes(include='number'))

    # Combine PCA with clusters
    df_combined = pd.concat([df_pca, df_clusters], axis=1)

    # Scatter clusters plot
    fig_scatter = _scatter_clusters(df_combined)
    outpath_scatter = base_dir / VIGNETTE_DIR / 'kscatter.html'
    fig_scatter.write_html(outpath_scatter)


def hierarchical_groups(height=None):
    """
    Public entry point expected by the pipeline/tests.
    Height parameter is accepted for compatibility with doit.
    """
    return hcluster_analysis()
