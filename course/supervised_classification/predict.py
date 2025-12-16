import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    pd.Series(y_pred).to_csv(y_pred_path, index=False)
    pd.DataFrame(y_prob).to_csv(y_pred_prob_path, index=False)


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
