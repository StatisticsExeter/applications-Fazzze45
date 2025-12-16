import pandas as pd
from sklearn.metrics import classification_report
from course.utils import find_project_root


def metric_report(y_test_path, y_pred_path, report_path):
    """Generate and save a classification report as a CSV file."""
    # Load the CSVs
    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)

    # Convert to string to avoid type mismatches
    y_test = y_test.astype(str).apply(lambda x: x.str.strip())
    y_pred = y_pred.astype(str).apply(lambda x: x.str.strip())

    # Generate the classification report
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_path, index=True)
    print(f"✅ Report saved to: {report_path}")


def metric_report_lda():
    """Run metrics for LDA classifier."""
    base_dir = find_project_root()
    y_test_path = base_dir / "data_cache" / "energy_y_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "lda_y_pred.csv"
    report_path = base_dir / "data_cache" / "vignettes" / "supervised_classification" / "lda.csv"
    metric_report(y_test_path, y_pred_path, report_path)


def metric_report_qda():
    """Run metrics for QDA classifier."""
    base_dir = find_project_root()
    y_test_path = base_dir / "data_cache" / "energy_y_test.csv"
    y_pred_path = base_dir / "data_cache" / "models" / "qda_y_pred.csv"
    report_path = base_dir / "data_cache" / "vignettes" / "supervised_classification" / "qda.csv"
    metric_report(y_test_path, y_pred_path, report_path)
