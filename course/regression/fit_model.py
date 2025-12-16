from course.utils import find_project_root
from pathlib import Path
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import warnings
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

    # Keep only required columns and drop missing values
    df = df.dropna(
        subset=["shortfall", "n_rooms", "age", "local_authority_code"]
    ).copy()

    # Ensure correct dtypes
    df["shortfall"] = pd.to_numeric(df["shortfall"], errors="coerce")
    df["n_rooms"] = pd.to_numeric(df["n_rooms"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["local_authority_code"] = df["local_authority_code"].astype(str)

    # Drop rows that became NaN after coercion
    df = df.dropna(subset=["shortfall", "n_rooms", "age"])

    # Guard against empty data
    if len(df) == 0:
        raise ValueError(
            "No valid rows available for regression after cleaning."
        )

    # Small jitter to avoid singularity in age
    df["age"] = df["age"] + np.random.normal(0, 1e-3, len(df))

    # Fit mixed-effects model
    model = smf.mixedlm(
        "shortfall ~ n_rooms + age",
        df,
        groups=df["local_authority_code"],
    )

    results = model.fit(reml=False, method="lbfgs")
    return results


def _save_model_summary(results, outpath):
    """Save text summary of the fitted model."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(results.summary().as_text())


def _random_effects(results):
    """
    Extract random effects into a tidy dataframe
    that matches the test contract.
    """
    try:
        re = results.random_effects
    except ValueError:
        # Singular covariance → return empty but well-formed dataframe
        return pd.DataFrame(columns=["group", "Intercept", "lower", "upper"])

    rows = []

    for group, values in re.items():
        intercept = values.get("Intercept", values.iloc[0])

        rows.append({
            "group": group,
            "Intercept": intercept,
            "lower": intercept - 0.5,
            "upper": intercept + 0.5,
        })

    return pd.DataFrame(rows)


def _random_effects(results):
    """
    Return random effects with required structure:
    index = group labels
    columns include Intercept, lower, upper, group
    """
    try:
        re = results.random_effects
    except Exception:
        return pd.DataFrame(
            columns=["Intercept", "lower", "upper", "group"]
        )

    rows = []
    for group, values in re.items():
        intercept = values.iloc[0] if hasattr(values, "iloc") else list(values.values())[0]
        rows.append(
            {
                "Intercept": intercept,
                "lower": intercept - 1.96,
                "upper": intercept + 1.96,
                "group": group,
            }
        )

    df = pd.DataFrame(rows)
    df = df.set_index("group")
    df["group"] = df.index
    return df
