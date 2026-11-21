import os
import sqlite3
import pandas as pd
from config import DB_PATH, BASE_DIR

PBI_DATA_DIR = os.path.join(BASE_DIR, "powerbi", "datasets")
os.makedirs(PBI_DATA_DIR, exist_ok=True)


def export_powerbi_tables():
    print("=" * 72)
    print("EXPORTING CLEAN STAR-SCHEMA TABLES FOR POWER BI DESKTOP")
    print("=" * 72)

    with sqlite3.connect(DB_PATH) as conn:
        # 1. SF Weekly Respiratory Fact Table (2025+ and full series)
        sf_fact = pd.read_sql_query(
            """
            SELECT
                w.week_start_date,
                w.week_end_date,
                w.year,
                w.month,
                w.year_month,
                w.epi_week,
                w.respiratory_virus,
                w.new_admissions,
                w.admission_rate_per_100k,
                w.sf_population_estimate,
                w.activity_level,
                w.trend,
                w.is_2025_plus,
                COALESCE(a.is_anomaly, 0) AS is_anomaly,
                COALESCE(a.anomaly_direction, 'Normal') AS anomaly_direction,
                COALESCE(a.z_score_4w, 0.0) AS z_score_4w
            FROM sf_respiratory_weekly w
            LEFT JOIN ml_anomalies a
              ON w.week_start_date = a.week_start_date
             AND w.respiratory_virus = a.respiratory_virus
            ORDER BY w.week_start_date, w.respiratory_virus
            """,
            conn,
        )

        # 2. ML Forecast Actual vs Predicted Table
        ml_forecast = pd.read_sql_query("SELECT * FROM ml_forecasts", conn)

        # 3. Next-Week Out-of-Sample Forecast Table
        ml_next = pd.read_sql_query("SELECT * FROM ml_next_week_forecast", conn)

        # 4. NHS England Provider Service Activity (Clean Provider Grain - No Double Counting)
        nhs_provider = pd.read_sql_query(
            """
            SELECT
                reporting_period_start,
                reporting_period_end,
                org_code,
                org_name,
                count_of,
                measure AS service_code,
                measure_desc AS service_type,
                COALESCE(measure_value, 0) AS total_activity,
                COALESCE(measure_value_0_18, 0) AS age_0_18,
                COALESCE(measure_value_19_64, 0) AS age_19_64,
                COALESCE(measure_value_65_plus, 0) AS age_65_plus
            FROM nhs_provider_activity
            WHERE org_level = 'Provider'
              AND dimension = 'ServiceType'
              AND measure_desc != 'Unspecified'
            """,
            conn,
        )

        # 5. NHS England Primary Referral Reasons (Clean Provider Grain)
        nhs_reasons = pd.read_sql_query(
            """
            SELECT
                reporting_period_start,
                reporting_period_end,
                org_code,
                org_name,
                count_of,
                measure AS reason_code,
                measure_desc AS primary_referral_reason,
                COALESCE(measure_value, 0) AS total_activity,
                COALESCE(measure_value_0_18, 0) AS age_0_18,
                COALESCE(measure_value_19_64, 0) AS age_19_64,
                COALESCE(measure_value_65_plus, 0) AS age_65_plus
            FROM nhs_provider_activity
            WHERE org_level = 'Provider'
              AND dimension = 'PrimaryReferralReason'
              AND measure_desc != 'Unspecified'
            """,
            conn,
        )

    exports = {
        "pbi_sf_respiratory_fact.csv": sf_fact,
        "pbi_ml_forecasts.csv": ml_forecast,
        "pbi_ml_next_week.csv": ml_next,
        "pbi_nhs_provider_services.csv": nhs_provider,
        "pbi_nhs_referral_reasons.csv": nhs_reasons,
    }

    for fname, df in exports.items():
        out_path = os.path.join(PBI_DATA_DIR, fname)
        df.to_csv(out_path, index=False)
        print(f"  Exported {fname:32s} -> {len(df):,} rows")

    print(f"\nAll Power BI tables saved in: {PBI_DATA_DIR}")


if __name__ == "__main__":
    export_powerbi_tables()