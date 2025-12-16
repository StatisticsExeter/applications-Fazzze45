from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
from course.utils import find_project_root


def fit_classifier(X_train_path, y_train_path, model_path, classifier):
    # Load training data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    # Fit model directly (NO renaming, NO cleaning)
    classifier.fit(X_train, y_train)

    # Save model
    joblib.dump(classifier, model_path)


def fit_lda():
    base_dir = find_project_root()
    X_train_path = base_dir / "data_cache" / "energy_X_train.csv"
    y_train_path = base_dir / "data_cache" / "energy_y_train.csv"
    model_path = base_dir / "data_cache" / "models" / "lda_model.joblib"

    classifier = DecisionTreeClassifier(random_state=42)
    fit_classifier(X_train_path, y_train_path, model_path, classifier)


def fit_qda():
    base_dir = find_project_root()
    X_train_path = base_dir / "data_cache" / "energy_X_train.csv"
    y_train_path = base_dir / "data_cache" / "energy_y_train.csv"
    model_path = base_dir / "data_cache" / "models" / "qda_model.joblib"

    classifier = DecisionTreeClassifier(random_state=42)
    fit_classifier(X_train_path, y_train_path, model_path, classifier)
