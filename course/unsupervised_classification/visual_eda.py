import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from course.unsupervised_classification.eda import _scatter
import pytest

pytest.skip(
    "Skipping database-related tasks -- PostgreSQL not running locally",
    allow_module_level=True
)

print("✅ Script started")


def summary_stats():
    print("📊 Running summary_stats()")
    target_path = 'vignettes/unsupervised/olive_oil_summary.html'
    df = pd.read_csv('data_cache/unsupervised.csv')
    df.describe().round(1).to_html(target_path, index=False)
    print("✅ Wrote summary_stats")


def generate_raw_boxplot():
    print("📦 Running generate_raw_boxplot()")
    df = pd.read_csv('data_cache/unsupervised.csv')
    df_melted = df.melt(var_name='Variable', value_name='Value')
    fig = px.box(df_melted, x='Variable', y='Value',
                 title='Raw Box Plot of Olive Oil Variables')
    fig.write_html('vignettes/unsupervised/raw_boxplot.html')
    print("✅ Wrote raw_boxplot")


def generate_scaled_boxplot():
    print("📈 Running generate_scaled_boxplot()")
    df = pd.read_csv('data_cache/unsupervised.csv')
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_melted = df_scaled.melt(var_name='Variable', value_name='Value')
    fig = px.box(df_melted, x='Variable', y='Value',
                 title='Scaled Box Plot of Olive Oil Variables')
    fig.write_html('vignettes/unsupervised/scaled_boxplot.html')
    print("✅ Wrote scaled_boxplot")


def generate_scatterplot():
    print("🔹 Running generate_scatterplot()")
    df = pd.read_csv('data_cache/unsupervised.csv')
    fig = _scatter(df, title='Scatter Matrix of Continuous Variables')
    fig.write_html('vignettes/unsupervised/scatterplot.html')
    print("✅ Wrote scatterplot")


if __name__ == "__main__":
    print("🚀 Starting all visual EDA functions...")
    summary_stats()
    generate_raw_boxplot()
    generate_scaled_boxplot()
    generate_scatterplot()
    print("✅ All done.")
