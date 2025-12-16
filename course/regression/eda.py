import pandas as pd
import plotly.express as px
from course.utils import find_project_root


def _boxplot(df, x_var, y_var, title):
    """Given a data frame 'df' containing categorical variable 'x_var'
    and outcome variable 'y_var', produce a box plot with all points shown"""
    fig = px.box(df, x=x_var, y=y_var, points='all', title=title)
    return fig


def _load_la_energy():
    base_dir = find_project_root()
    path = base_dir / "data_cache" / "la_energy.csv"
    return pd.read_csv(path)


def boxplot_age():
    """
    Make a boxplot of shortfall vs age and save to vignettes/regression/boxplot_age.html
    """
    df = _load_la_energy()
    fig = _boxplot(df, "age", "shortfall", "Shortfall by Age Category")

    out_path = find_project_root() / "vignettes" / "regression" / "boxplot_age.html"
    fig.write_html(out_path)


def boxplot_rooms():
    """
    Optional: boxplot of shortfall vs n_rooms
    """
    df = _load_la_energy()
    fig = _boxplot(df, "n_rooms", "shortfall", "Shortfall by Number of Rooms")

    out_path = find_project_root() / "vignettes" / "regression" / "boxplot_rooms.html"
    fig.write_html(out_path)
