"""
dashboard.py — Interactive Streamlit dashboard for biomarker method comparison.

Usage:
    streamlit run src/dashboard.py

Reads results/dashboard/{disease}_dashboard_data.csv (built by build_dashboard_data.py).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_DIR = Path("results")
DASHBOARD_DIR = RESULTS_DIR / "dashboard"

st.set_page_config(page_title="Biomarker Pipeline Dashboard", layout="wide")
st.title("Biomarker Formula Comparison Dashboard")

# ── Sidebar controls ──────────────────────────────────────────────────────────
available_diseases = sorted(
    p.stem.replace("_dashboard_data", "")
    for p in DASHBOARD_DIR.glob("*_dashboard_data.csv")
) if DASHBOARD_DIR.exists() else []

if not available_diseases:
    st.error(
        "No dashboard data found. Run:\n"
        "```\npython src/build_dashboard_data.py --disease ra\n```"
    )
    st.stop()

disease = st.sidebar.selectbox("Disease", available_diseases, index=0)
data_path = DASHBOARD_DIR / f"{disease}_dashboard_data.csv"
df = pd.read_csv(data_path)

method_options = sorted(df["method"].unique())
selected_methods = st.sidebar.multiselect("Methods", method_options, default=method_options)

filtered = df[df["method"].isin(selected_methods)].copy()

# ── Metric cards ──────────────────────────────────────────────────────────────
st.subheader(f"Best AUC-PR per method — {disease.upper()}")
cols = st.columns(len(method_options))
for i, m in enumerate(method_options):
    sub = df[df["method"] == m]
    if not sub.empty:
        best = sub.loc[sub["auc_pr"].idxmax()]
        cols[i].metric(
            label=m.upper(),
            value=f"{best['auc_pr']:.4f}",
            delta=f"ROC {best['auc_roc']:.4f}",
        )

# ── Bar chart ────────────────────────────────────────────────────────────────
st.subheader("AUC-PR by Method and Variant")
if not filtered.empty:
    fig = px.bar(
        filtered.sort_values("auc_pr", ascending=False),
        x="variant", y="auc_pr", color="method",
        text="auc_pr", hover_data=["formula_display", "auc_roc"],
        labels={"auc_pr": "AUC-PR", "variant": "Variant"},
        height=420,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

# ── Formula table ─────────────────────────────────────────────────────────────
st.subheader("Formula Details")
display_cols = ["method", "variant", "auc_pr", "auc_roc", "formula_display"]
st.dataframe(
    filtered[display_cols].sort_values("auc_pr", ascending=False).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)
