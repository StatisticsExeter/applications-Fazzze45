from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
from course.utils import find_project_root


def fit_classifier(X_train_path, y_train_path, model_path, classifier):
    # Load training data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path, header=None).squeeze()

    # Rename feature columns
    X_train.columns = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Combine X and y to keep alignment
    df = X_train.copy()
    df["target"] = y_train

    # Drop rows ONLY if target is missing
    df = df.dropna(subset=["target"])

    # Convert features to numeric
    feature_cols = df.columns.difference(["target"])
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Simple imputation: fill missing features with column mean
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # Reset index
    df = df.reset_index(drop=True)

    # Split back
    X_clean = df[feature_cols]
    y_clean = df["target"]

    # Final safety check
    if len(X_clean) == 0:
        raise ValueError("No training samples available after cleaning.")

    classifier.fit(X_clean, y_clean)
    joblib.dump(classifier, model_path)


def fit_lda():
    base_dir = find_project_root()
    X_train_path = base_dir / "data_cache" / "energy_X_train.csv"
    y_train_path = base_dir / "data_cache" / "energy_y_train.csv"
    model_path = base_dir / "data_cache" / "lda_model.joblib"

    classifier = DecisionTreeClassifier(random_state=42)
    fit_classifier(X_train_path, y_train_path, model_path, classifier)


def fit_qda():
    base_dir = find_project_root()
    X_train_path = base_dir / "data_cache" / "energy_X_train.csv"
    y_train_path = base_dir / "data_cache" / "energy_y_train.csv"
    model_path = base_dir / "data_cache" / "qda_model.joblib"

    classifier = DecisionTreeClassifier(random_state=42)
    fit_classifier(X_train_path, y_train_path, model_path, classifier)