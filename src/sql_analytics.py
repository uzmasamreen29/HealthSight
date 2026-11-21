import os
import json
import sqlite3
import pandas as pd
from config import DB_PATH, PROCESSED_DIR

SQL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.sql")
KPI_JSON_PATH = os.path.join(PROCESSED_DIR, "kpi_summary.json")


def load_named_queries(sql_path: str = SQL_FILE_PATH) -> dict[str, str]:
    """Parse -- QUERY: <name> blocks from analytics.sql."""
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    queries = {}
    current_name = None
    buffer = []

    for line in content.splitlines():
        if line.strip().startswith("-- QUERY:"):
            if current_name and buffer:
                queries[current_name] = "\n".join(buffer).strip()
            current_name = line.split("-- QUERY:", 1)[1].strip()
            buffer = []
        else:
            if current_name is not None:
                buffer.append(line)

    if current_name and buffer:
        queries[current_name] = "\n".join(buffer).strip()

    return queries


def execute_all_analytics() -> dict[str, pd.DataFrame]:
    """Run all named SQL queries and return results as DataFrames."""
    queries = load_named_queries()
    results = {}

    with sqlite3.connect(DB_PATH) as conn:
        for name, sql in queries.items():
            results[name] = pd.read_sql_query(sql, conn)

    return results


def build_and_save_kpi_summary(results: dict[str, pd.DataFrame]) -> dict:
    """Create a verified JSON snapshot of KPIs for Gemini AI and Streamlit."""
    sf_share = results["sf_disease_share_and_ranking"].to_dict(orient="records")
    sf_recent = results["sf_weekly_trends_and_wow"].head(9).to_dict(orient="records")
    nhs_top_providers = results["nhs_top_providers_activity"].head(5).to_dict(orient="records")
    nhs_top_services = results["nhs_service_type_demographics"].head(5).to_dict(orient="records")
    provenance = results["data_provenance_audit"].to_dict(orient="records")

    summary = {
        "provenance": provenance,
        "sf_disease_summary_2025_plus": sf_share,
        "sf_latest_3_weeks_by_virus": sf_recent,
        "nhs_top_5_providers_july_2026": nhs_top_providers,
        "nhs_top_5_services_july_2026": nhs_top_services,
    }

    with open(KPI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("=" * 72)
    print("RUNNING HEALTHSIGHT ADVANCED SQL ANALYTICS")
    print("=" * 72)

    dfs = execute_all_analytics()
    for q_name, df in dfs.items():
        print(f"\n--- [{q_name}] ({len(df)} rows) ---")
        print(df.head(5).to_string(index=False))

    build_and_save_kpi_summary(dfs)
    print(f"\nSaved verified KPI JSON snapshot to: {KPI_JSON_PATH}")