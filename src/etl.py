import os
import sqlite3
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

from config import (
    DB_PATH,
    SF_RESPIRATORY_URL,
    SF_RAW_CSV,
    NHS_ZIP_PATH,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HealthSight-ETL/1.0"
}


def fetch_and_clean_sf_data(force_download: bool = True) -> tuple[pd.DataFrame, dict]:
    """Download, validate, and clean San Francisco Weekly Respiratory Admissions."""
    print("[1/3] Processing San Francisco Respiratory Virus Hospital Admissions...")
    source_mode = "live_api"

    if force_download or not os.path.exists(SF_RAW_CSV):
        try:
            resp = requests.get(SF_RESPIRATORY_URL, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            with open(SF_RAW_CSV, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            if os.path.exists(SF_RAW_CSV):
                print(f"  Warning: Live download failed ({e}); using cached {SF_RAW_CSV}")
                source_mode = "local_cache"
            else:
                raise RuntimeError(f"Failed to download SF dataset and no cache exists: {e}")
    else:
        source_mode = "local_cache"

    df = pd.read_csv(SF_RAW_CSV)
    initial_rows = len(df)

    # Standardize dates
    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    df["week_end_date"] = pd.to_datetime(df["week_end_date"], errors="coerce")
    df = df.dropna(subset=["week_start_date", "respiratory_virus"]).copy()

    # Clean numeric fields
    df["sf_population_estimate"] = (
        df["sf_population_estimate"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["sf_population_estimate"] = pd.to_numeric(df["sf_population_estimate"], errors="coerce").astype("Int64")
    df["new_admissions"] = pd.to_numeric(df["new_admissions"], errors="coerce").fillna(0).astype(int)
    df["admission_rate_per_100k"] = pd.to_numeric(df["admission_rate_per_100k"], errors="coerce").fillna(0.0)
    df["epi_week"] = pd.to_numeric(df["epi_week"], errors="coerce").astype("Int64")

    # Standardize string categories
    for col in ["respiratory_virus", "activity_level", "trend"]:
        df[col] = df[col].astype(str).str.strip()

    # Derived calendar columns
    df["year"] = df["week_start_date"].dt.year
    df["month"] = df["week_start_date"].dt.month
    df["year_month"] = df["week_start_date"].dt.strftime("%Y-%m")
    df["is_2025_plus"] = (df["week_start_date"] >= "2025-01-01").astype(int)

    # Deduplicate on natural composite key
    df = df.sort_values(["week_start_date", "respiratory_virus"]).drop_duplicates(
        subset=["week_start_date", "respiratory_virus"], keep="last"
    )

    # Format dates as ISO strings for clean SQLite storage
    min_date = df["week_start_date"].min().strftime("%Y-%m-%d")
    max_date = df["week_end_date"].max().strftime("%Y-%m-%d")
    df["week_start_date"] = df["week_start_date"].dt.strftime("%Y-%m-%d")
    df["week_end_date"] = df["week_end_date"].dt.strftime("%Y-%m-%d")

    keep_cols = [
        "week_start_date",
        "week_end_date",
        "year",
        "month",
        "year_month",
        "epi_week",
        "respiratory_virus",
        "new_admissions",
        "admission_rate_per_100k",
        "sf_population_estimate",
        "activity_level",
        "trend",
        "is_2025_plus",
        "data_updated_at",
        "data_as_of",
    ]
    df = df[keep_cols].reset_index(drop=True)

    meta = {
        "dataset_key": "sf_respiratory_weekly",
        "dataset_name": "San Francisco Respiratory Virus Hospital Admissions",
        "source_url": SF_RESPIRATORY_URL,
        "ingestion_mode": source_mode,
        "raw_rows": initial_rows,
        "clean_rows": len(df),
        "rows_2025_plus": int(df["is_2025_plus"].sum()),
        "earliest_observation": min_date,
        "latest_observation": max_date,
    }
    print(f"  Cleaned {len(df):,} rows ({meta['rows_2025_plus']:,} from 2025+) | Range: {min_date} to {max_date}")
    return df, meta


def clean_and_load_nhs_data(conn: sqlite3.Connection) -> dict:
    """Read, validate, and load NHS England July 2026 ZIP dataset into SQLite in chunks."""
    print("[2/3] Processing NHS England July 2026 Care Activity Dataset...")
    if not os.path.exists(NHS_ZIP_PATH):
        raise FileNotFoundError(
            f"NHS dataset not found at {NHS_ZIP_PATH}. Place csds-jul26-exp-core-data.zip there first."
        )

    table_name = "nhs_provider_activity"
    conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    total_raw = 0
    total_clean = 0
    min_period = None
    max_period = None

    numeric_cols = [
        "measure_value",
        "measure_value_0_18",
        "measure_value_19_64",
        "measure_value_65_plus",
    ]

    chunk_iter = pd.read_csv(NHS_ZIP_PATH, chunksize=50000, low_memory=False)
    for idx, chunk in enumerate(chunk_iter):
        total_raw += len(chunk)
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Standardize string columns
        str_cols = [
            "reporting_period_start",
            "reporting_period_end",
            "org_code",
            "org_name",
            "org_level",
            "dimension",
            "count_of",
            "measure",
            "measure_desc",
            "measure_2",
            "measure_2_desc",
        ]
        for col in str_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna("Unspecified").astype(str).str.strip()

        # Convert numeric columns (NHS uses '*' or NaN for suppressed small counts)
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.drop_duplicates()
        total_clean += len(chunk)

        c_min = chunk["reporting_period_start"].min()
        c_max = chunk["reporting_period_end"].max()
        if min_period is None or c_min < min_period:
            min_period = c_min
        if max_period is None or c_max > max_period:
            max_period = c_max

        if_exists_mode = "replace" if idx == 0 else "append"
        chunk.to_sql(table_name, conn, if_exists=if_exists_mode, index=False)

    # Create analytical indexes for fast SQL & Streamlit queries
    conn.execute(f"CREATE INDEX idx_nhs_org_level ON {table_name} (org_level, org_code);")
    conn.execute(f"CREATE INDEX idx_nhs_dimension ON {table_name} (dimension, count_of);")
    conn.execute(f"CREATE INDEX idx_nhs_measure ON {table_name} (measure, measure_desc);")

    meta = {
        "dataset_key": "nhs_provider_activity",
        "dataset_name": "NHS England Community Services Activity (July 2026)",
        "source_url": "https://digital.nhs.uk (csds-jul26-exp-core-data.zip)",
        "ingestion_mode": "official_archive",
        "raw_rows": total_raw,
        "clean_rows": total_clean,
        "rows_2025_plus": total_clean,
        "earliest_observation": min_period,
        "latest_observation": max_period,
    }
    print(f"  Loaded {total_clean:,} clean rows | Period: {min_period} to {max_period}")
    return meta


def run_etl(force_download: bool = True) -> dict:
    """Execute full end-to-end ETL pipeline and update SQLite database."""
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sf_df, sf_meta = fetch_and_clean_sf_data(force_download=force_download)

    with sqlite3.connect(DB_PATH) as conn:
        # Load SF Respiratory data
        sf_df.to_sql("sf_respiratory_weekly", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX idx_sf_virus_date ON sf_respiratory_weekly (respiratory_virus, week_start_date);"
        )

        # Load NHS England data
        nhs_meta = clean_and_load_nhs_data(conn)

        # Record audit metadata
        print("[3/3] Updating data_ingestion_log metadata table...")
        sf_meta["retrieved_at_utc"] = retrieved_at
        nhs_meta["retrieved_at_utc"] = retrieved_at

        log_df = pd.DataFrame([sf_meta, nhs_meta])
        log_df.to_sql("data_ingestion_log", conn, if_exists="replace", index=False)
        conn.commit()

    print(f"\nETL COMPLETE -> SQLite database saved at: {DB_PATH}")
    return {"sf": sf_meta, "nhs": nhs_meta, "db_path": DB_PATH}


if __name__ == "__main__":
    run_etl(force_download=True)