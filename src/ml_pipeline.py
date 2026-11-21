import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import DB_PATH, MODELS_DIR


def load_sf_series() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                week_start_date,
                week_end_date,
                year,
                month,
                epi_week,
                respiratory_virus,
                new_admissions,
                admission_rate_per_100k,
                activity_level,
                trend,
                is_2025_plus
            FROM sf_respiratory_weekly
            ORDER BY respiratory_virus, week_start_date
            """,
            conn,
        )
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create strictly lagged time-series features per virus (zero data leakage)."""
    frames = []
    for virus, grp in df.groupby("respiratory_virus", sort=False):
        g = grp.sort_values("week_start_date").copy()

        # Lagged admission counts
        g["lag_1"] = g["new_admissions"].shift(1)
        g["lag_2"] = g["new_admissions"].shift(2)
        g["lag_3"] = g["new_admissions"].shift(3)
        g["lag_4"] = g["new_admissions"].shift(4)

        # Lagged rolling statistics (computed strictly up to t-1)
        g["rolling_mean_4w"] = g["new_admissions"].shift(1).rolling(4, min_periods=2).mean()
        g["rolling_std_4w"] = g["new_admissions"].shift(1).rolling(4, min_periods=2).std().fillna(0.0)
        g["wow_diff_lag1"] = g["lag_1"] - g["lag_2"]
        g["rate_lag_1"] = g["admission_rate_per_100k"].shift(1)

        # Cyclical seasonality features
        g["sin_week"] = np.sin(2 * np.pi * g["epi_week"] / 52.0)
        g["cos_week"] = np.cos(2 * np.pi * g["epi_week"] / 52.0)

        frames.append(g)

    feat_df = pd.concat(frames, ignore_index=True)
    feat_df = feat_df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_4"]).reset_index(drop=True)

    # One-hot encode respiratory_virus
    virus_dummies = pd.get_dummies(feat_df["respiratory_virus"], prefix="virus", dtype=float)
    feat_df = pd.concat([feat_df, virus_dummies], axis=1)
    return feat_df


def run_forecasting(feat_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Task A: Chronological forecasting of weekly respiratory admissions."""
    unique_weeks = sorted(feat_df["week_start_date"].unique())
    split_idx = int(len(unique_weeks) * 0.80)
    cutoff_week = unique_weeks[split_idx]

    train_df = feat_df[feat_df["week_start_date"] < cutoff_week].copy()
    test_df = feat_df[feat_df["week_start_date"] >= cutoff_week].copy()

    X_train, y_train = train_df[feature_cols], train_df["new_admissions"]
    X_test, y_test = test_df[feature_cols], test_df["new_admissions"]

    # Model 1: Persistence Baseline (t-1 actual)
    baseline_preds = np.maximum(0, test_df["lag_1"].values)

    # Model 2: Ridge Regression
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0, random_state=42)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_preds = np.maximum(0, ridge_pipe.predict(X_test))

    # Model 3: Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    rf_reg.fit(X_train, y_train)
    rf_preds = np.maximum(0, rf_reg.predict(X_test))

    def eval_reg(y_true, y_pred):
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 3),
            "r2": round(float(r2_score(y_true, y_pred)), 3),
        }

    metrics = {
        "Persistence_Baseline": eval_reg(y_test, baseline_preds),
        "Ridge_Regression": eval_reg(y_test, ridge_preds),
        "Random_Forest_Regressor": eval_reg(y_test, rf_preds),
    }

    # Select best model by lowest MAE
    best_model_name = min(metrics, key=lambda k: metrics[k]["mae"])
    best_model_obj = rf_reg if best_model_name == "Random_Forest_Regressor" else ridge_pipe
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "admission_forecaster.joblib"))

    # Build chronological predictions table across full dataset for dashboarding
    all_rf_preds = np.round(np.maximum(0, rf_reg.predict(feat_df[feature_cols])), 2)
    all_ridge_preds = np.round(np.maximum(0, ridge_pipe.predict(feat_df[feature_cols])), 2)

    forecast_table = feat_df[
        ["week_start_date", "week_end_date", "respiratory_virus", "new_admissions", "admission_rate_per_100k"]
    ].copy()
    forecast_table["split"] = np.where(feat_df["week_start_date"] < cutoff_week, "Train", "Test")
    forecast_table["rf_predicted_admissions"] = all_rf_preds
    forecast_table["ridge_predicted_admissions"] = all_ridge_preds
    forecast_table["baseline_predicted_admissions"] = feat_df["lag_1"].values

    # Generate next-week out-of-sample forecast per virus from latest observed row
    raw_df = load_sf_series()
    next_week_forecasts = []
    for virus, grp in raw_df.groupby("respiratory_virus"):
        g = grp.sort_values("week_start_date").tail(5).reset_index(drop=True)
        last_row = g.iloc[-1]
        next_start = (pd.to_datetime(last_row["week_start_date"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        next_end = (pd.to_datetime(last_row["week_end_date"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        next_epi = int(last_row["epi_week"]) + 1 if int(last_row["epi_week"]) < 52 else 1

        row_dict = {
            "lag_1": float(g.iloc[-1]["new_admissions"]),
            "lag_2": float(g.iloc[-2]["new_admissions"]),
            "lag_3": float(g.iloc[-3]["new_admissions"]),
            "lag_4": float(g.iloc[-4]["new_admissions"]),
            "rolling_mean_4w": float(g.iloc[-4:]["new_admissions"].mean()),
            "rolling_std_4w": float(g.iloc[-4:]["new_admissions"].std()),
            "wow_diff_lag1": float(g.iloc[-1]["new_admissions"] - g.iloc[-2]["new_admissions"]),
            "rate_lag_1": float(g.iloc[-1]["admission_rate_per_100k"]),
            "sin_week": float(np.sin(2 * np.pi * next_epi / 52.0)),
            "cos_week": float(np.cos(2 * np.pi * next_epi / 52.0)),
            "virus_COVID-19": 1.0 if virus == "COVID-19" else 0.0,
            "virus_Influenza": 1.0 if virus == "Influenza" else 0.0,
            "virus_RSV": 1.0 if virus == "RSV" else 0.0,
        }
        X_next = pd.DataFrame([row_dict])[feature_cols]
        pred_rf = round(float(max(0.0, rf_reg.predict(X_next)[0])), 2)
        pred_ridge = round(float(max(0.0, ridge_pipe.predict(X_next)[0])), 2)

        next_week_forecasts.append({
            "respiratory_virus": virus,
            "forecast_week_start": next_start,
            "forecast_week_end": next_end,
            "last_observed_admissions": int(last_row["new_admissions"]),
            "rf_next_week_forecast": pred_rf,
            "ridge_next_week_forecast": pred_ridge,
        })

    return {
        "train_weeks": int(train_df["week_start_date"].nunique()),
        "test_weeks": int(test_df["week_start_date"].nunique()),
        "test_start_week": cutoff_week,
        "best_model": best_model_name,
        "metrics": metrics,
        "next_week_forecasts": next_week_forecasts,
        "forecast_table": forecast_table,
    }


def run_trend_classification(feat_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Task B: Classify weekly disease trend (Increasing, Decreasing, Stable)."""
    valid_df = feat_df[feat_df["trend"].isin(["Increasing", "Decreasing", "Stable"])].copy()
    unique_weeks = sorted(valid_df["week_start_date"].unique())
    cutoff_week = unique_weeks[int(len(unique_weeks) * 0.80)]

    train_df = valid_df[valid_df["week_start_date"] < cutoff_week]
    test_df = valid_df[valid_df["week_start_date"] >= cutoff_week]

    X_train, y_train = train_df[feature_cols], train_df["trend"]
    X_test, y_test = test_df[feature_cols], test_df["trend"]

    models = {
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 3),
            "precision_weighted": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 3),
            "recall_weighted": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 3),
            "f1_weighted": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 3),
        }
        if name == "Random_Forest_Classifier":
            joblib.dump(clf, os.path.join(MODELS_DIR, "trend_classifier.joblib"))

    return {
        "classes": sorted(valid_df["trend"].unique().tolist()),
        "test_samples": len(test_df),
        "metrics": results,
    }


def run_anomaly_detection(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Task C: Detect unusual hospital admission spikes/drops per virus using Isolation Forest."""
    anomaly_frames = []
    summary_by_virus = {}

    for virus, grp in feat_df.groupby("respiratory_virus", sort=False):
        g = grp.sort_values("week_start_date").copy()
        g["z_score_4w"] = np.where(
            g["rolling_std_4w"] > 0,
            (g["new_admissions"] - g["rolling_mean_4w"]) / g["rolling_std_4w"],
            0.0,
        )
        g["wow_change"] = g["new_admissions"] - g["lag_1"]

        iso_features = g[["new_admissions", "wow_change", "z_score_4w"]].fillna(0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(iso_features)

        iso = IsolationForest(n_estimators=200, contamination=0.06, random_state=42)
        raw_pred = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        g["is_anomaly"] = np.where(raw_pred == -1, 1, 0)
        g["anomaly_score"] = np.round(scores, 4)
        g["anomaly_direction"] = np.where(
            g["is_anomaly"] == 0,
            "Normal",
            np.where(g["wow_change"] >= 0, "Surge Spike", "Sharp Drop"),
        )

        flagged = g[g["is_anomaly"] == 1]
        summary_by_virus[virus] = {
            "total_weeks": int(len(g)),
            "anomalies_detected": int(len(flagged)),
            "flagged_weeks": flagged[
                ["week_start_date", "new_admissions", "wow_change", "anomaly_direction"]
            ].to_dict(orient="records"),
        }
        anomaly_frames.append(g)

    out_df = pd.concat(anomaly_frames, ignore_index=True)[
        [
            "week_start_date",
            "week_end_date",
            "respiratory_virus",
            "new_admissions",
            "rolling_mean_4w",
            "wow_change",
            "z_score_4w",
            "is_anomaly",
            "anomaly_score",
            "anomaly_direction",
        ]
    ]
    out_df["rolling_mean_4w"] = out_df["rolling_mean_4w"].round(2)
    out_df["z_score_4w"] = out_df["z_score_4w"].round(2)

    return out_df, summary_by_virus


def run_ml_pipeline() -> dict:
    print("=" * 72)
    print("RUNNING HEALTHSIGHT MACHINE LEARNING PIPELINE")
    print("=" * 72)

    raw_df = load_sf_series()
    feat_df = build_feature_matrix(raw_df)

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "rolling_mean_4w",
        "rolling_std_4w",
        "wow_diff_lag1",
        "rate_lag_1",
        "sin_week",
        "cos_week",
        "virus_COVID-19",
        "virus_Influenza",
        "virus_RSV",
    ]

    # Task A: Forecasting
    forecast_res = run_forecasting(feat_df, feature_cols)
    forecast_table = forecast_res.pop("forecast_table")

    # Task B: Trend Classification
    trend_res = run_trend_classification(feat_df, feature_cols)

    # Task C: Anomaly Detection
    anomaly_df, anomaly_summary = run_anomaly_detection(feat_df)

    # Persist ML outputs to SQLite for Power BI & Streamlit
    with sqlite3.connect(DB_PATH) as conn:
        forecast_table.to_sql("ml_forecasts", conn, if_exists="replace", index=False)
        anomaly_df.to_sql("ml_anomalies", conn, if_exists="replace", index=False)
        pd.DataFrame(forecast_res["next_week_forecasts"]).to_sql(
            "ml_next_week_forecast", conn, if_exists="replace", index=False
        )
        conn.commit()

    report = {
        "forecasting": forecast_res,
        "trend_classification": trend_res,
        "anomaly_detection": anomaly_summary,
    }

    report_path = os.path.join(MODELS_DIR, "ml_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[A] Hospital Admission Forecasting (Chronological Test Set):")
    for m_name, m_vals in forecast_res["metrics"].items():
        print(f"  {m_name:25s} -> MAE: {m_vals['mae']:6.3f} | RMSE: {m_vals['rmse']:6.3f} | R2: {m_vals['r2']:6.3f}")
    print("\n  Next-Week Out-of-Sample Forecasts:")
    for item in forecast_res["next_week_forecasts"]:
        print(
            f"  {item['respiratory_virus']:10s} ({item['forecast_week_start']}): "
            f"Last={item['last_observed_admissions']} | RF={item['rf_next_week_forecast']} | Ridge={item['ridge_next_week_forecast']}"
        )

    print("\n[B] Disease Trend Classification (Chronological Test Set):")
    for m_name, m_vals in trend_res["metrics"].items():
        print(
            f"  {m_name:25s} -> Acc: {m_vals['accuracy']:.3f} | Prec: {m_vals['precision_weighted']:.3f} | "
            f"Rec: {m_vals['recall_weighted']:.3f} | F1: {m_vals['f1_weighted']:.3f}"
        )

    print("\n[C] Isolation Forest Anomaly Detection Summary:")
    for virus, info in anomaly_summary.items():
        print(f"  {virus:10s} -> {info['anomalies_detected']} anomalies out of {info['total_weeks']} weeks")

    print(f"\nML outputs saved to SQLite & {report_path}")
    return report


if __name__ == "__main__":
    run_ml_pipeline()