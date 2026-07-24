from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from course.utils import find_project_root


VIGNETTE_DIR = (
    Path("data_cache")
    / "vignettes"
    / "unsupervised_classification"
)


def compare_clustering_methods() -> None:
    """Compare K-Means and complete-linkage agglomerative clustering."""

    base_dir = find_project_root()
    input_path = base_dir / "data_cache" / "la_collision.csv"
    output_dir = base_dir / VIGNETTE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)

    authority_names = data["lad_ons"].astype(str)
    numeric_data = data.select_dtypes(include="number")

    if len(numeric_data) < 3:
        raise ValueError(
            "At least three observations are required for this comparison."
        )

    scaled_data = StandardScaler().fit_transform(numeric_data)

    # Two clusters are used because the dataset contains only three observations.
    kmeans_model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )
    kmeans_labels = kmeans_model.fit_predict(scaled_data)

    agglomerative_model = AgglomerativeClustering(
        n_clusters=2,
        linkage="complete",
    )
    agglomerative_labels = agglomerative_model.fit_predict(scaled_data)

    kmeans_silhouette = silhouette_score(
        scaled_data,
        kmeans_labels,
    )
    agglomerative_silhouette = silhouette_score(
        scaled_data,
        agglomerative_labels,
    )

    comparison = pd.DataFrame(
        {
            "Local Authority": authority_names,
            "K-Means Cluster": kmeans_labels,
            "Agglomerative Cluster": agglomerative_labels,
        }
    )

    comparison.to_csv(
        output_dir / "cluster_comparison.csv",
        index=False,
    )

    scores = pd.DataFrame(
        {
            "Method": ["K-Means", "Agglomerative"],
            "Silhouette Score": [
                kmeans_silhouette,
                agglomerative_silhouette,
            ],
        }
    )

    scores.to_csv(
        output_dir / "cluster_validation_scores.csv",
        index=False,
    )

    print("\nCluster assignments:")
    print(comparison.to_string(index=False))

    print("\nSilhouette scores:")
    print(scores.to_string(index=False))

    return None
