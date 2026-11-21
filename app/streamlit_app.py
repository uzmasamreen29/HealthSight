import os
import sys
import json
import sqlite3
import pandas as pd
import streamlit as st

# Add src/ to Python path so modules import cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DB_PATH, PROCESSED_DIR, MODELS_DIR, GEMINI_API_KEY
from etl import run_etl
from sql_analytics import execute_all_analytics, build_and_save_kpi_summary
from ml_pipeline import run_ml_pipeline
from export_powerbi import export_powerbi_tables
from genai_assistant import ask_healthsight_assistant

st.set_page_config(
    page_title="HealthSight | GenAI Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_dashboard_data():
    with sqlite3.connect(DB_PATH) as conn:
        provenance = pd.read_sql_query("SELECT * FROM data_ingestion_log", conn)
        sf_df = pd.read_sql_query("SELECT * FROM sf_respiratory_weekly ORDER BY week_start_date", conn)
        forecasts_df = pd.read_sql_query("SELECT * FROM ml_forecasts ORDER BY week_start_date", conn)
        next_week_df = pd.read_sql_query("SELECT * FROM ml_next_week_forecast", conn)
        anomalies_df = pd.read_sql_query("SELECT * FROM ml_anomalies ORDER BY week_start_date DESC", conn)
        nhs_providers = pd.read_sql_query(
            """
            SELECT
                org_code,
                org_name,
                SUM(COALESCE(measure_value, 0)) AS total_contacts,
                SUM(COALESCE(measure_value_0_18, 0)) AS age_0_18,
                SUM(COALESCE(measure_value_19_64, 0)) AS age_19_64,
                SUM(COALESCE(measure_value_65_plus, 0)) AS age_65_plus
            FROM nhs_provider_activity
            WHERE org_level = 'Provider'
              AND dimension = 'ServiceType'
              AND count_of = 'UniqueCareContacts'
            GROUP BY org_code, org_name
            ORDER BY total_contacts DESC
            LIMIT 15
            """,
            conn,
        )
        nhs_services = pd.read_sql_query(
            """
            SELECT
                measure_desc AS service_type,
                SUM(COALESCE(measure_value, 0)) AS total_contacts,
                SUM(COALESCE(measure_value_0_18, 0)) AS age_0_18,
                SUM(COALESCE(measure_value_19_64, 0)) AS age_19_64,
                SUM(COALESCE(measure_value_65_plus, 0)) AS age_65_plus
            FROM nhs_provider_activity
            WHERE org_level = 'Provider'
              AND dimension = 'ServiceType'
              AND count_of = 'UniqueCareContacts'
              AND measure_desc != 'Unspecified'
            GROUP BY measure_desc
            ORDER BY total_contacts DESC
            LIMIT 12
            """,
            conn,
        )

    ml_report = {}
    report_path = os.path.join(MODELS_DIR, "ml_evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            ml_report = json.load(f)

    return {
        "provenance": provenance,
        "sf": sf_df,
        "forecasts": forecasts_df,
        "next_week": next_week_df,
        "anomalies": anomalies_df,
        "nhs_providers": nhs_providers,
        "nhs_services": nhs_services,
        "ml_report": ml_report,
    }


# Sidebar: Controls, Refresh & Provenance
with st.sidebar:
    st.title("🏥 HealthSight Controls")
    st.caption("Healthcare Analytics & Hospital Intelligence")

    user_api_key = st.text_input(
        "Gemini API Key (Optional Override)",
        type="password",
        placeholder="Uses .env GEMINI_API_KEY if blank",
    )

    st.divider()
    st.subheader("🔄 Live Data Pipeline")
    if st.button("Refresh Live Datasets & Retrain ML", use_container_width=True):
        with st.spinner("Fetching live San Francisco data, updating SQLite, retraining ML & exporting Power BI CSVs..."):
            run_etl(force_download=True)
            sql_res = execute_all_analytics()
            build_and_save_kpi_summary(sql_res)
            run_ml_pipeline()
            export_powerbi_tables()
            st.cache_data.clear()
        st.success("Pipeline refreshed and models updated!")
        st.rerun()

    st.divider()
    st.subheader("📋 Data Source Provenance")
    if os.path.exists(DB_PATH):
        data = load_dashboard_data()
        for _, row in data["provenance"].iterrows():
            st.markdown(f"**{row['dataset_name']}**")
            st.caption(
                f"Rows: {row['clean_rows']:,} | Window: {row['earliest_observation']} → {row['latest_observation']}\n"
                f"Retrieved: {row['retrieved_at_utc']}"
            )
    else:
        st.error("Database not found. Click Refresh above or run `python src/etl.py`.")
        st.stop()


# Main Header
st.title("🏥 HealthSight – GenAI-Powered Healthcare Analytics & Hospital Intelligence")
st.markdown(
    "Real-world surveillance & operations intelligence combining **San Francisco Weekly Respiratory Virus Hospital Admissions (2024–2026)** "
    "and **NHS England Provider Care Activity (July 2026)**."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Healthcare Overview",
    "2. Disease Trends & NHS Ops",
    "3. Admission Forecasting",
    "4. Anomaly Detection",
    "5. Model Evaluation",
    "6. Gemini AI Assistant",
])

# -------------------------------------------------------------------------
# TAB 1: HEALTHCARE OVERVIEW
# -------------------------------------------------------------------------
with tab1:
    sf_2025 = data["sf"][data["sf"]["is_2025_plus"] == 1].copy()
    total_adm = int(sf_2025["new_admissions"].sum())
    avg_rate = round(float(sf_2025["admission_rate_per_100k"].mean()), 2)
    spike_weeks = int(data["anomalies"][data["anomalies"]["is_anomaly"] == 1].shape[0])
    nhs_contacts = int(data["nhs_services"]["total_contacts"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SF Respiratory Admissions (2025+)", f"{total_adm:,}")
    c2.metric("Avg Weekly Rate / 100k (2025+)", f"{avg_rate}")
    c3.metric("Detected Anomaly Weeks", f"{spike_weeks}")
    c4.metric("NHS Top-12 Services Contacts (Jul 2026)", f"{nhs_contacts:,}")

    st.subheader("Weekly Respiratory Hospital Admissions (San Francisco, 2025–2026)")
    pivot_weekly = sf_2025.pivot(
        index="week_start_date", columns="respiratory_virus", values="new_admissions"
    )
    st.line_chart(pivot_weekly)

    st.subheader("Disease-Wise Burden Comparison (2025–2026)")
    virus_summary = (
        sf_2025.groupby("respiratory_virus", as_index=False)
        .agg(
            Total_Admissions=("new_admissions", "sum"),
            Avg_Weekly_Admissions=("new_admissions", "mean"),
            Peak_Weekly_Admissions=("new_admissions", "max"),
            Avg_Rate_Per_100k=("admission_rate_per_100k", "mean"),
        )
        .sort_values("Total_Admissions", ascending=False)
    )
    virus_summary["Avg_Weekly_Admissions"] = virus_summary["Avg_Weekly_Admissions"].round(2)
    virus_summary["Avg_Rate_Per_100k"] = virus_summary["Avg_Rate_Per_100k"].round(2)
    st.dataframe(virus_summary, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# TAB 2: DISEASE TRENDS & NHS HOSPITAL OPERATIONS
# -------------------------------------------------------------------------
with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("San Francisco Disease Activity Explorer")
        selected_virus = st.selectbox(
            "Select Respiratory Virus",
            sorted(data["sf"]["respiratory_virus"].unique()),
            key="trend_virus",
        )
        v_df = data["sf"][data["sf"]["respiratory_virus"] == selected_virus].copy()
        st.area_chart(v_df.set_index("week_start_date")[["admission_rate_per_100k"]])
        st.dataframe(
            v_df[
                [
                    "week_start_date",
                    "week_end_date",
                    "new_admissions",
                    "admission_rate_per_100k",
                    "activity_level",
                    "trend",
                ]
            ].sort_values("week_start_date", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=280,
        )

    with col_right:
        st.subheader("NHS England Provider & Demographic Activity (July 2026)")
        st.caption("Source: NHS England Community Services Data Set (UniqueCareContacts, Provider Grain)")
        st.bar_chart(
            data["nhs_services"].set_index("service_type")[["age_0_18", "age_19_64", "age_65_plus"]]
        )
        st.dataframe(data["nhs_providers"], use_container_width=True, hide_index=True, height=280)

# -------------------------------------------------------------------------
# TAB 3: ADMISSION FORECASTING
# -------------------------------------------------------------------------
with tab3:
    st.subheader("Next-Week Out-of-Sample Admission Forecasts")
    st.dataframe(data["next_week"], use_container_width=True, hide_index=True)

    st.subheader("Actual vs. Predicted Weekly Admissions (Chronological Train/Test)")
    f_virus = st.selectbox(
        "Filter Forecast by Virus",
        sorted(data["forecasts"]["respiratory_virus"].unique()),
        key="forecast_virus",
    )
    f_sub = data["forecasts"][data["forecasts"]["respiratory_virus"] == f_virus].copy()
    chart_df = f_sub.set_index("week_start_date")[
        ["new_admissions", "rf_predicted_admissions", "ridge_predicted_admissions"]
    ].rename(
        columns={
            "new_admissions": "Actual Admissions",
            "rf_predicted_admissions": "Random Forest Forecast",
            "ridge_predicted_admissions": "Ridge Regression Forecast",
        }
    )
    st.line_chart(chart_df)

# -------------------------------------------------------------------------
# TAB 4: ANOMALY DETECTION
# -------------------------------------------------------------------------
with tab4:
    st.subheader("Isolation Forest Hospital Admission Anomaly Detection")
    anom_only = st.checkbox("Show Flagged Anomaly Weeks Only", value=True)
    anom_view = data["anomalies"].copy()
    if anom_only:
        anom_view = anom_view[anom_view["is_anomaly"] == 1]

    st.dataframe(anom_view, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# TAB 5: MODEL EVALUATION REPORTS
# -------------------------------------------------------------------------
with tab5:
    st.subheader("Machine Learning Model Evaluation Benchmarks")
    ml_rep = data["ml_report"]
    if ml_rep:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**A. Hospital Admission Forecasting (Chronological Test Split)**")
            reg_df = pd.DataFrame(ml_rep["forecasting"]["metrics"]).T.reset_index()
            reg_df.columns = ["Model", "MAE", "RMSE", "R2 Score"]
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
            st.info(f"Selected Best MAE Model: **{ml_rep['forecasting']['best_model']}**")

        with c_right:
            st.markdown("**B. Disease Trend Classification (Increasing / Decreasing / Stable)**")
            clf_df = pd.DataFrame(ml_rep["trend_classification"]["metrics"]).T.reset_index()
            clf_df.columns = ["Model", "Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
            st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.markdown("**C. Isolation Forest Anomaly Summary**")
        anom_cards = []
        for virus, info in ml_rep["anomaly_detection"].items():
            anom_cards.append({
                "Virus": virus,
                "Total Evaluated Weeks": info["total_weeks"],
                "Flagged Anomaly Weeks": info["anomalies_detected"],
            })
        st.dataframe(pd.DataFrame(anom_cards), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# TAB 6: GEMINI HEALTHCARE ANALYTICS ASSISTANT
# -------------------------------------------------------------------------
with tab6:
    st.subheader("🤖 Gemini Healthcare Analytics AI Assistant")
    st.caption(
        "Grounded strictly in verified HealthSight SQL KPIs and ML outputs. "
        "All operational insights are data-driven suggestions, not medical advice."
    )

    quick_prompts = [
        "Summarise the latest San Francisco weekly respiratory admission trends and next-week forecasts.",
        "Explain the unusual admission spikes detected by Isolation Forest and suggest operational capacity considerations.",
        "Compare COVID-19, Influenza, and RSV burden from 2025 onward using verified KPIs.",
        "Summarise NHS England July 2026 top service lines and 65+ demographic demand patterns.",
    ]

    selected_preset = st.selectbox("Quick Operational Questions:", ["Custom Question..."] + quick_prompts)
    default_text = "" if selected_preset == "Custom Question..." else selected_preset

    user_q = st.text_area("Ask a question about HealthSight analytics:", value=default_text, height=100)
    if st.button("Generate Grounded Operational Insight", type="primary"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Analyzing verified SQL & ML metrics with Gemini..."):
                effective_key = user_api_key.strip() if user_api_key.strip() else GEMINI_API_KEY
                answer = ask_healthsight_assistant(user_q, api_key=effective_key)
            st.markdown(answer)