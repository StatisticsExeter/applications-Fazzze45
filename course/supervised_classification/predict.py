import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    # Load model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    # 🔧 Ensure feature names match training
    X_test.columns = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Make predictions
    y_pred = model.predict(X_test)

    # Try to get prediction probabilities
    try:
        y_pred_prob = model.predict_proba(X_test)
        if y_pred_prob.ndim == 2:
            y_pred_prob = y_pred_prob[:, 1]
    except AttributeError:
        y_pred_prob = [0.5] * len(y_pred)

    # Save outputs
    pd.Series(y_pred, name="predicted_built_age").to_csv(y_pred_path, index=False)
    pd.Series(
        y_pred_prob, name="predicted_built_age_prob"
    ).to_csv(y_pred_prob_path, index=False)


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "lda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "lda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "lda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / "data_cache" / "models" / "qda_model.joblib"
    X_test_path = base_dir / "data_cache" / "energy_X_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "qda_y_pred.csv"
    y_pred_prob_path = base_dir / "data_cache" / "models" / "qda_y_pred_prob.csv"
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
