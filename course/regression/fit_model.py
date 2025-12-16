from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence expected numerical warnings from MixedLM
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "regression"


def _fit_model(df):
    """
    Fit a linear mixed-effects model with:
    - shortfall as response
    - n_rooms and age as fixed effects
    - local_authority_code as random intercept
    """

    # Keep required columns and drop missing values
    df = df.dropna(
        subset=["shortfall", "n_rooms", "age", "local_authority_code"]
    ).copy()

    # Ensure numeric types
    df["shortfall"] = pd.to_numeric(df["shortfall"], errors="coerce")
    df["n_rooms"] = pd.to_numeric(df["n_rooms"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["local_authority_code"] = df["local_authority_code"].astype(str)

    # Drop rows that became NaN
    df = df.dropna(subset=["shortfall", "n_rooms", "age"])

    if len(df) == 0:
        raise ValueError("No valid rows available for regression.")

    # Add tiny jitter to avoid singular matrix
    df["age"] = df["age"] + np.random.normal(0, 1e-3, len(df))

    model = smf.mixedlm(
        "shortfall ~ n_rooms + age",
        df,
        groups=df["local_authority_code"],
    )

    results = model.fit(reml=False, method="lbfgs")
    return results


def _random_effects(results):
    """
    Extract random intercepts and compute confidence intervals.
    Output must match test contract exactly.
    """
    re = results.random_effects

    # Convert to DataFrame
    re_df = pd.DataFrame.from_dict(re, orient="index")

    # Ensure Intercept exists
    if "Intercept" not in re_df.columns:
        re_df["Intercept"] = re_df.iloc[:, 0]

    # Standard error from covariance matrix
    stderr = float(np.sqrt(results.cov_re.iloc[0, 0]))

    # Confidence intervals
    re_df["lower"] = re_df["Intercept"] - 1.96 * stderr
    re_df["upper"] = re_df["Intercept"] + 1.96 * stderr

    # Add group as a COLUMN (not index)
    re_df["group"] = re_df.index

    # Reset index to default integer index
    re_df = re_df.reset_index(drop=True)

    # Return columns in required order
    return re_df[["Intercept", "group", "lower", "upper"]]


def _save_model_summary(results, outpath):
    """Save text summary of the fitted model."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(results.summary().as_text())


def fit_model():
    base_dir = Path(".")
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")

    results = _fit_model(df)

    # Save summary
    _save_model_summary(
        results,
        VIGNETTE_DIR / "model_fit.txt",
    )

    # Save random effects
    re_df = _random_effects(results)
    re_df.to_csv(
        base_dir / "data_cache" / "models" / "reffs.csv"
    )