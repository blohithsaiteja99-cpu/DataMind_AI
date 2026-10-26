import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from dataminds.analysis import (
    build_investigation,
    detect_anomalies,
    investigate_root_causes,
    numeric_correlations,
    profile_dataset,
    run_hypothesis_test,
)
from dataminds.ai import explain_findings, ollama_status
from dataminds.io import read_dataset
from dataminds.report import markdown_report

st.set_page_config(
    page_title="DataMind AI — Autonomous Data Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #090d16;
    --bg-card: rgba(15, 23, 42, 0.75);
    --border-card: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(16, 185, 129, 0.35);
    --accent-emerald: #10b981;
    --accent-cyan: #06b6d4;
    --accent-indigo: #6366f1;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}

.hero-container {
    position: relative;
    padding: 1.75rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(13, 78, 74, 0.55) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 350px;
    height: 100%;
    background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.15), transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #ffffff 30%, #34d399 70%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
    font-weight: 400;
    letter-spacing: -0.01em;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
}

.status-badge.emerald {
    border-color: rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    background: rgba(6, 78, 59, 0.35);
}

.status-badge.cyan {
    border-color: rgba(6, 182, 212, 0.3);
    color: #67e8f9;
    background: rgba(8, 51, 68, 0.35);
}

.status-badge.amber {
    border-color: rgba(245, 158, 11, 0.3);
    color: #fcd34d;
    background: rgba(69, 26, 3, 0.35);
}

.pulsing-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.75; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.75; }
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(16, 185, 129, 0.35);
    box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.15);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--kpi-accent, #10b981), transparent);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
}

.kpi-icon {
    font-size: 1.15rem;
    opacity: 0.85;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-weight: 500;
}

.custom-alert {
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    backdrop-filter: blur(10px);
}

.custom-alert.info {
    border-left: 4px solid #06b6d4;
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.08) 0%, rgba(15, 23, 42, 0.5) 100%);
}

.custom-alert.warning {
    border-left: 4px solid #f59e0b;
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, rgba(15, 23, 42, 0.5) 100%);
}

.custom-alert.success {
    border-left: 4px solid #10b981;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.5) 100%);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(15, 23, 42, 0.5);
    padding: 0.4rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 2.6rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #94a3b8;
    padding: 0 1rem;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background: rgba(255, 255, 255, 0.04);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
}

[data-testid="stSidebar"] {
    background: #090e18;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    border-color: rgba(16, 185, 129, 0.4);
}
</style>
""",
    unsafe_allow_html=True,
)


def apply_plotly_theme(fig, height=400):
    """Applies high-end dark aesthetics to any Plotly chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=12),
        margin=dict(t=45, b=40, l=45, r=25),
        title_font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color="#f8fafc"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(color="#cbd5e1"),
        ),
    )
    return fig


def render_kpi(title, value, subtext, icon, accent="#10b981"):
    """Generates an HTML glass KPI card."""
    return f"""
    <div class="kpi-card" style="--kpi-accent: {accent};">
        <div class="kpi-header">
            <span class="kpi-title">{title}</span>
            <span class="kpi-icon">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """


with st.sidebar:
    st.markdown("### ⚙️ Investigation Setup")
    st.caption("Load data, configure models, and target metrics.")

    uploaded = st.file_uploader(
        "Upload Dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Supports files up to 25 MB with automatic schema discovery.",
    )
    demo = st.checkbox(
        "Use demo restaurant dataset",
        value=uploaded is None,
        help="Loads sample restaurant metrics with built-in anomalies and missing values.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Local AI (Ollama)")
    ollama_model = st.text_input("Ollama Model Tag", value="llama3.2")
    status = ollama_status()

    if status.get("available"):
        installed_models = status.get("models", [])
        if installed_models:
            st.markdown(
                f"""
            <div class="status-badge emerald" style="width: 100%; justify-content: flex-start; margin-bottom: 0.5rem;">
                <span class="pulsing-dot"></span>
                <span>Ollama Online ({len(installed_models)} model{'s' if len(installed_models) > 1 else ''} ready)</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            picked = st.selectbox("Detected local models", options=installed_models, index=0)
            if picked:
                ollama_model = picked
        else:
            st.markdown(
                """
            <div class="status-badge emerald" style="width: 100%; justify-content: flex-start; margin-bottom: 0.5rem;">
                <span class="pulsing-dot"></span>
                <span>Ollama Service Connected</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.caption("⚠️ No models installed yet. Run in terminal: `ollama pull llama3.2`")
    else:
        st.markdown(
            """
        <div class="status-badge amber" style="width: 100%; justify-content: flex-start; margin-bottom: 0.5rem;">
            <span>⚠️</span>
            <span>Ollama Offline (LLM optional)</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.caption("Statistical analytics, anomalies, and tests work 100% offline.")

    st.markdown("---")
    st.caption(
        "🔒 **Privacy Guarantee**: All computations are performed locally on your machine. No data is transmitted externally."
    )


dataset_name = "sample_restaurant_metrics.csv"
df = None

try:
    if uploaded is not None:
        df = read_dataset(uploaded.name, uploaded.getvalue())
        dataset_name = uploaded.name
    elif demo:
        sample_path = Path(__file__).resolve().parent / "data" / "sample_restaurant_metrics.csv"
        df = pd.read_csv(sample_path)
        dataset_name = "sample_restaurant_metrics.csv"
    else:
        st.info("👈 Upload a dataset or enable the sample dataset in the sidebar to begin.")
        st.stop()
except Exception as exc:
    st.error(f"Failed to read dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("The selected dataset is empty.")
    st.stop()

numeric_cols = list(df.select_dtypes(include=np.number).columns)
categorical_cols = list(df.select_dtypes(exclude=np.number).columns)


hero_html = f"""
<div class="hero-container">
    <h1 class="hero-title">🧠 DataMind AI</h1>
    <p class="hero-subtitle">Autonomous Investigation Engine · Rigorous Statistical Evidence · Explanatory AI</p>
    <div class="badge-row">
        <div class="status-badge emerald">
            <span class="pulsing-dot"></span>
            <span>Dataset: <strong>{dataset_name}</strong></span>
        </div>
        <div class="status-badge cyan">
            <span>📊 {len(df):,} Rows</span>
        </div>
        <div class="status-badge cyan">
            <span>🏷️ {len(df.columns)} Columns ({len(numeric_cols)} Numeric, {len(categorical_cols)} Categorical)</span>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    target_options = ["None (General Profiling)"] + numeric_cols
    selected_target = st.selectbox(
        "🎯 Primary Investigation Target Metric (Optional)",
        target_options,
        index=1 if numeric_cols else 0,
        help="Select a key numeric KPI (e.g. revenue, sales, churn score) to uncover low vs. high driver associations.",
    )
    target = None if selected_target.startswith("None") else selected_target

with top_c2:
    st.write("")
    st.write("")
    with st.popover("🔍 Preview Raw Records"):
        st.caption(f"Showing first 100 rows of {dataset_name}")
        st.dataframe(df.head(100), width="stretch")


with st.spinner("Executing statistical profiling, Isolation Forest, and pattern analysis..."):
    findings = build_investigation(df, target=target)

p = findings["profile"]
a = findings["anomalies"]


kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        render_kpi("Observations", f"{p['rows']:,}", f"{p['duplicate_rows']} duplicates", "📋", "#10b981"),
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    st.markdown(
        render_kpi(
            "Features",
            f"{p['columns_count']}",
            f"{len(numeric_cols)} num · {len(categorical_cols)} cat",
            "🏷️",
            "#06b6d4",
        ),
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    completeness = round(100.0 - float(p["missing_percent_overall"]), 2)
    st.markdown(
        render_kpi(
            "Completeness",
            f"{completeness}%",
            f"{p['missing_cells']:,} missing cells",
            "🛡️",
            "#6366f1",
        ),
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    anomaly_cnt = a.get("anomaly_count", 0)
    anomaly_pct = a.get("anomaly_percent", 0.0)
    st.markdown(
        render_kpi("Flagged Anomalies", f"{anomaly_cnt:,}", f"{anomaly_pct}% of total records", "🚨", "#f43f5e"),
        unsafe_allow_html=True,
    )

st.write("")


tabs = st.tabs(
    [
        "📊 Overview & Trends",
        "🛡️ Data Quality",
        "🚨 Anomaly Detection",
        "🎯 Root-Cause Candidates",
        "⚖️ Hypothesis Testing",
        "🧠 AI Executive Briefing",
        "📄 Investigation Report",
    ]
)


with tabs[0]:
    st.markdown("#### 📈 Distribution & Correlation Intelligence")

    col_dist, col_corr = st.columns(2)

    with col_dist:
        st.markdown("##### 📦 Numeric Feature Distribution")
        if numeric_cols:
            dist_col = st.selectbox("Select feature to inspect", numeric_cols, key="ov_dist_col")
            chart_type = st.radio(
                "Display Type",
                ["Histogram + Box marginal", "Violin Plot", "Cumulative ECDF"],
                horizontal=True,
                key="ov_chart_type",
            )

            if chart_type == "Histogram + Box marginal":
                fig_dist = px.histogram(
                    df,
                    x=dist_col,
                    marginal="box",
                    nbins=35,
                    color_discrete_sequence=["#10b981"],
                    opacity=0.85,
                )
            elif chart_type == "Violin Plot":
                fig_dist = px.violin(
                    df,
                    y=dist_col,
                    box=True,
                    points="outliers",
                    color_discrete_sequence=["#06b6d4"],
                )
            else:
                fig_dist = px.ecdf(
                    df,
                    x=dist_col,
                    color_discrete_sequence=["#6366f1"],
                )

            apply_plotly_theme(fig_dist, height=360)
            st.plotly_chart(fig_dist, width="stretch")

            s = df[dist_col].dropna()
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            stat_c1.metric("Mean", f"{s.mean():,.2f}")
            stat_c2.metric("Median", f"{s.median():,.2f}")
            stat_c3.metric("Std Dev", f"{s.std():,.2f}")
            stat_c4.metric("Range", f"[{s.min():,.2f}, {s.max():,.2f}]")
        else:
            st.info("No numeric features detected in this dataset.")

    with col_corr:
        st.markdown("##### 🌐 Interactive Correlation Matrix")
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols].corr(numeric_only=True)
            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Tealrose",
                zmin=-1,
                zmax=1,
            )
            apply_plotly_theme(fig_corr, height=360)
            st.plotly_chart(fig_corr, width="stretch")
        else:
            st.info("At least two numeric features are required to compute correlations.")

    st.markdown("---")
    st.markdown("##### 🔗 Strongest Pairwise Correlations & Scatter Explorer")

    corrs = findings.get("correlations", [])
    if corrs:
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.caption("Top ranked statistical relationships sorted by absolute correlation coefficient:")
            corr_table = pd.DataFrame(corrs)
            corr_table.columns = ["Feature A", "Feature B", "Correlation (r)", "Strength"]
            st.dataframe(corr_table, width="stretch", hide_index=True)

        with c_right:
            pair_options = [f"{c['feature_a']} vs {c['feature_b']} (r = {c['correlation']})" for c in corrs[:8]]
            chosen_pair_idx = st.selectbox(
                "Explore Pairwise Scatter Plot",
                range(len(pair_options)),
                format_func=lambda i: pair_options[i],
            )
            pair_data = corrs[chosen_pair_idx]
            fa, fb = pair_data["feature_a"], pair_data["feature_b"]

            color_var = target if target and target in df.columns else None
            fig_scatter = px.scatter(
                df,
                x=fa,
                y=fb,
                color=color_var,
                color_continuous_scale="Viridis",
                opacity=0.75,
                title=f"{fa} vs {fb} (Pearson r = {pair_data['correlation']})",
            )
            valid_pts = df[[fa, fb]].dropna()
            if len(valid_pts) >= 2:
                from scipy import stats as sp_stats
                lr = sp_stats.linregress(valid_pts[fa], valid_pts[fb])
                x_vals = np.linspace(valid_pts[fa].min(), valid_pts[fa].max(), 50)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=lr.slope * x_vals + lr.intercept,
                        mode="lines",
                        name=f"Trendline (slope: {lr.slope:.2f})",
                        line=dict(color="#10b981", dash="dash", width=2),
                    )
                )
            apply_plotly_theme(fig_scatter, height=330)
            st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("No numeric correlation pairs found.")


with tabs[1]:
    st.markdown("#### 🛡️ Automated Data Quality & Schema Integrity")

    q_col1, q_col2 = st.columns([1, 2])

    with q_col1:
        st.markdown("##### 🩺 Quality Checks & Diagnostics")
        for note in p.get("quality_notes", []):
            if "duplicate" in note.lower() or "missing values" in note.lower():
                st.markdown(
                    f"""
                <div class="custom-alert warning">
                    <span>⚠️</span>
                    <span style="font-size: 0.88rem; color: #cbd5e1;">{note}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif "no major issues" in note.lower():
                st.markdown(
                    f"""
                <div class="custom-alert success">
                    <span>✅</span>
                    <span style="font-size: 0.88rem; color: #cbd5e1;">{note}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="custom-alert info">
                    <span>ℹ️</span>
                    <span style="font-size: 0.88rem; color: #cbd5e1;">{note}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "DataMind AI operates in read-only audit mode — data is examined without silent mutations or dropping."
        )

    with q_col2:
        st.markdown("##### 📋 Column Schema & Distribution Health")
        col_profile_data = []
        for c in p.get("columns", []):
            stats_dict = c.get("statistics")
            mean_val = f"{stats_dict['mean']:,.2f}" if stats_dict and stats_dict.get("mean") is not None else "-"
            range_val = (
                f"[{stats_dict['min']:,.2f}, {stats_dict['max']:,.2f}]"
                if stats_dict and stats_dict.get("min") is not None
                else "-"
            )
            col_profile_data.append(
                {
                    "Feature": c["name"],
                    "Type": c["dtype"],
                    "Missing": f"{c['missing_count']} ({c['missing_percent']}%)",
                    "Distinct Values": c["unique_count"],
                    "Mean / Baseline": mean_val,
                    "Observed Range": range_val,
                }
            )

        col_df = pd.DataFrame(col_profile_data)
        st.dataframe(col_df, width="stretch", hide_index=True)


with tabs[2]:
    st.markdown("#### 🚨 Isolation Forest Outlier & Anomaly Detection")

    anom_c1, anom_c2 = st.columns([1, 2])

    with anom_c1:
        st.markdown("##### 🎛️ Anomaly Sensitivity")
        contamination_rate = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Proportion of expected anomalies in the dataset (default: 5%).",
        )

        if st.button("⚡ Re-run Anomaly Detection", width="stretch"):
            with st.spinner("Fitting Isolation Forest ensemble..."):
                st.session_state["anomalies"] = detect_anomalies(df, contamination=contamination_rate)

        current_anomalies = st.session_state.get("anomalies", a)

        st.markdown(
            f"""
        <div class="custom-alert info" style="margin-top: 1rem;">
            <span>ℹ️</span>
            <div style="font-size: 0.85rem; color: #cbd5e1;">
                <strong>Method:</strong> {current_anomalies.get('method', 'Isolation Forest')}<br>
                <strong>Flagged Rows:</strong> {current_anomalies.get('anomaly_count', 0)} ({current_anomalies.get('anomaly_percent', 0.0)}%)<br>
                <em>{current_anomalies.get('note', '')}</em>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with anom_c2:
        st.markdown("##### 🎯 2D Anomaly Projection Map")
        flagged_records = current_anomalies.get("rows", [])
        if len(numeric_cols) >= 2:
            sc_x = st.selectbox("X-Axis Feature", numeric_cols, index=0, key="anom_x")
            sc_y = st.selectbox("Y-Axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="anom_y")

            anom_indices = {r["row_index"] for r in flagged_records}
            plot_df = df.copy()
            plot_df["Status"] = ["Anomaly" if i in anom_indices else "Normal" for i in range(len(plot_df))]

            fig_anom = px.scatter(
                plot_df,
                x=sc_x,
                y=sc_y,
                color="Status",
                color_discrete_map={"Normal": "#06b6d4", "Anomaly": "#f43f5e"},
                opacity=0.8,
                title=f"Isolation Forest Flags across {sc_x} vs {sc_y}",
            )
            apply_plotly_theme(fig_anom, height=340)
            st.plotly_chart(fig_anom, width="stretch")
        else:
            st.info("At least two numeric features are required for 2D anomaly plotting.")

    st.markdown("---")
    st.markdown("##### 📑 Highest-Scoring Flagged Anomalies")
    if flagged_records:
        adf = pd.DataFrame(flagged_records)
        st.dataframe(adf, width="stretch", hide_index=True)

        csv_bytes = adf.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Flagged Anomalies CSV",
            data=csv_bytes,
            file_name=f"{dataset_name.split('.')[0]}_anomalies.csv",
            mime="text/csv",
        )
    else:
        st.info("No anomalous records detected with current sensitivity settings.")


with tabs[3]:
    st.markdown("#### 🎯 Driver & Root-Cause Association Analysis")
    st.caption("Ranks features showing significant median shifts between bottom 20% and top 20% outcome cohorts.")

    if numeric_cols:
        rc_target = st.selectbox(
            "Select Target Metric for Cohort Analysis",
            numeric_cols,
            index=numeric_cols.index(target) if target in numeric_cols else 0,
            key="rc_target_select",
        )

        if st.button("🔍 Investigate Candidate Factors", width="stretch"):
            with st.spinner("Conducting Mann–Whitney U hypothesis tests across cohorts..."):
                st.session_state["root_cause"] = investigate_root_causes(df, rc_target)

        rc_res = st.session_state.get("root_cause", findings.get("root_cause_investigation", {}))

        if rc_res.get("status") == "ok":
            st.markdown(
                f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
                <div class="status-badge amber">
                    <span>🔻 Low Cohort: {rc_res.get('low_group_definition', '')}</span>
                </div>
                <div class="status-badge emerald">
                    <span>🔺 High Cohort: {rc_res.get('high_group_definition', '')}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            candidates = rc_res.get("candidates", [])
            if candidates:
                cand_df = pd.DataFrame(candidates)

                c_chart, c_dive = st.columns([3, 2])

                with c_chart:
                    cand_df["Direction"] = [
                        "Higher in High Cohort" if v > 0 else "Lower in High Cohort"
                        for v in cand_df["standardized_median_difference"]
                    ]
                    fig_bar = px.bar(
                        cand_df.head(10),
                        x="standardized_median_difference",
                        y="feature",
                        orientation="h",
                        color="Direction",
                        color_discrete_map={
                            "Higher in High Cohort": "#10b981",
                            "Lower in High Cohort": "#f43f5e",
                        },
                        title="Standardized Median Shift (High Cohort minus Low Cohort)",
                    )
                    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
                    apply_plotly_theme(fig_bar, height=380)
                    st.plotly_chart(fig_bar, width="stretch")

                with c_dive:
                    st.markdown("##### 🔬 Feature Cohort Inspection")
                    cand_feature = st.selectbox(
                        "Deep dive into feature distribution",
                        options=cand_df["feature"].tolist(),
                    )

                    y_series = pd.to_numeric(df[rc_target], errors="coerce")
                    q20, q80 = y_series.quantile([0.2, 0.8])
                    sub_df = df.copy()
                    sub_df["Cohort"] = "Middle 60%"
                    sub_df.loc[y_series <= q20, "Cohort"] = "Low Cohort (Bottom 20%)"
                    sub_df.loc[y_series >= q80, "Cohort"] = "High Cohort (Top 20%)"
                    sub_df = sub_df[sub_df["Cohort"] != "Middle 60%"]

                    fig_cohort = px.box(
                        sub_df,
                        x="Cohort",
                        y=cand_feature,
                        color="Cohort",
                        color_discrete_map={
                            "Low Cohort (Bottom 20%)": "#f59e0b",
                            "High Cohort (Top 20%)": "#10b981",
                        },
                        points="all",
                    )
                    apply_plotly_theme(fig_cohort, height=320)
                    st.plotly_chart(fig_cohort, width="stretch")

                st.markdown("##### 📊 Detailed Candidate Metrics Table")
                st.dataframe(cand_df, width="stretch", hide_index=True)
            else:
                st.info("No significant candidate associations met the threshold.")

            st.markdown(
                f"""
            <div class="custom-alert info" style="margin-top: 1rem;">
                <span>⚠️</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;">{rc_res.get('interpretation', '')}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            msg = rc_res.get("message", "Click 'Investigate Candidate Factors' to begin.")
            st.info(msg)
    else:
        st.info("A numeric target metric is required for root-cause analysis.")


with tabs[4]:
    st.markdown("#### ⚖️ Evidence-Based Hypothesis Testing")
    st.caption("Rigorous statistical group comparisons: Welch's independent t-test (2 groups) or One-Way ANOVA (>2 groups).")

    if numeric_cols and len(df.columns) > 1:
        ht_c1, ht_c2 = st.columns(2)
        with ht_c1:
            test_target = st.selectbox("Numeric Outcome Metric (Target)", numeric_cols, key="ht_num_target")
        with ht_c2:
            group_options = [c for c in df.columns if c != test_target]
            test_group = st.selectbox("Grouping Factor (Categorical Column)", group_options, key="ht_group_col")

        if st.button("🧪 Execute Hypothesis Test", width="stretch"):
            try:
                with st.spinner("Computing statistical test and effect size..."):
                    st.session_state["hyp_test"] = run_hypothesis_test(df, test_target, test_group)
            except ValueError as val_err:
                st.error(str(val_err))

        h_result = st.session_state.get("hyp_test")
        if h_result:
            p_val = h_result.get("p_value")
            is_significant = p_val is not None and p_val < 0.05

            st.markdown("---")

            verdict_badge = "emerald" if is_significant else "amber"
            verdict_text = (
                "Statistically Significant Difference (p < 0.05)"
                if is_significant
                else "Insufficient Evidence of Group Difference (p ≥ 0.05)"
            )
            st.markdown(
                f"""
            <div class="custom-alert {'success' if is_significant else 'info'}">
                <span style="font-size: 1.3rem;">{'🎯' if is_significant else 'ℹ️'}</span>
                <div>
                    <strong style="color: #ffffff; font-size: 1rem;">{h_result.get('test', 'Statistical Test')}</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.88rem;">{h_result.get('interpretation', '')}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("p-Value", f"{p_val:.5f}" if p_val is not None else "N/A")
            m_c2.metric("Test Statistic", f"{h_result.get('statistic', 0.0):.4f}")
            eff_key = "effect_size_cohens_d_approx" if "effect_size_cohens_d_approx" in h_result else "effect_size_eta_squared"
            eff_name = "Cohen's d (approx)" if "cohen" in eff_key else "Eta²"
            eff_val = h_result.get(eff_key)
            m_c3.metric(eff_name, f"{eff_val:.4f}" if eff_val is not None else "N/A")
            m_c4.metric("Groups Tested", f"{len(h_result.get('groups', []))}")

            v_col, t_col = st.columns([3, 2])
            with v_col:
                clean_test_df = df[[test_target, test_group]].dropna()
                fig_hyp = px.violin(
                    clean_test_df,
                    x=test_group,
                    y=test_target,
                    color=test_group,
                    box=True,
                    points="all",
                    title=f"Distribution of {test_target} by {test_group}",
                )
                apply_plotly_theme(fig_hyp, height=360)
                st.plotly_chart(fig_hyp, width="stretch")

            with t_col:
                st.markdown("##### 👥 Group Sample Statistics")
                g_df = pd.DataFrame(h_result.get("groups", []))
                st.dataframe(g_df, width="stretch", hide_index=True)

                st.markdown(
                    f"""
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 1rem; border-left: 2px solid #64748b; padding-left: 0.75rem;">
                    <strong>Null Hypothesis:</strong> {h_result.get('null_hypothesis', '')}<br>
                    <em>{h_result.get('caution', '')}</em>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Need at least one numeric outcome and one separate grouping column to run hypothesis tests.")


with tabs[5]:
    st.markdown("#### 🧠 Local AI Executive Briefing")
    st.caption("Synthesizes computed statistics, anomalies, and driver associations into an executive narrative.")

    ai_top_left, ai_top_right = st.columns([3, 1])

    with ai_top_left:
        st.write(
            "The LLM provides narrative interpretation of **already verified Python calculations**. It does not perform internal math or guess numbers."
        )

    with ai_top_right:
        if st.button("✨ Generate AI Explanation", width="stretch"):
            with st.spinner("Synthesizing findings via local Ollama engine..."):
                st.session_state["ai"] = explain_findings(findings, model=ollama_model)

    ai_state = st.session_state.get("ai")
    if ai_state:
        if ai_state.get("available"):
            st.markdown(
                f"""
            <div class="custom-alert success">
                <span>🤖</span>
                <div>
                    <strong>Executive Synthesis Completed (Model: {ai_state.get('model', 'Ollama')})</strong>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1.5rem 1.75rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); margin-top: 1rem;">
                {ai_state.get('text', '')}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="custom-alert warning">
                <span>⚠️</span>
                <div>
                    <strong>{ai_state.get('text', 'Ollama Request Failed')}</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">{ai_state.get('details', '')}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if "not installed" in ai_state.get("text", "").lower() or "empty" in ai_state.get("details", "").lower():
                st.markdown(
                    f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-top: 0.5rem;">
                    <strong>To enable AI explanations, open PowerShell and download the model:</strong>
                    <pre style="margin-top: 0.5rem; background: #0b0f19; padding: 0.75rem; border-radius: 8px;"><code>ollama pull {ollama_model}</code></pre>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Click 'Generate AI Explanation' above to request an automated briefing from your local LLM.")


with tabs[6]:
    st.markdown("#### 📄 Executive Investigation Report")
    st.caption("Export publication-ready Markdown reports and JSON data findings.")

    rep_content = markdown_report(findings, dataset_name)

    btn_c1, btn_c2, _ = st.columns([1, 1, 2])
    with btn_c1:
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=rep_content.encode("utf-8"),
            file_name=f"{dataset_name.split('.')[0]}_investigation_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with btn_c2:
        findings_json = json.dumps(findings, indent=2, default=str)
        st.download_button(
            label="⬇️ Download Findings JSON",
            data=findings_json.encode("utf-8"),
            file_name=f"{dataset_name.split('.')[0]}_findings.json",
            mime="application/json",
            width="stretch",
        )

    st.markdown("---")
    st.markdown(
        f"""
    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem 2rem;">
        {rep_content}
    </div>
    """,
        unsafe_allow_html=True,
    )
