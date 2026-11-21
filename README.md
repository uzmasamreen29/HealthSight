# HealthSight – GenAI-Powered Healthcare Analytics & Hospital Intelligence

An end-to-end healthcare intelligence platform that ingests real-world 2025–2026 public health datasets, stores structured analytics in SQLite, executes advanced SQL window/CTE queries, trains machine learning forecasting, classification, and anomaly detection models, and delivers grounded operational insights via Google Gemini API, Power BI, and Streamlit.

## Architecture

```text
HealthSight/
├── data/
│   ├── raw/                  # Raw downloaded sources & verification metadata
│   └── processed/            # SQLite DB (healthsight.db) & verified KPI JSON
├── src/
│   ├── config.py             # Paths, URLs, and Gemini model config
│   ├── verify_sources.py     # Live schema & date-range inspection
│   ├── etl.py                # Automated ingestion, cleaning, validation & SQLite load
│   ├── analytics.sql         # CTEs, window functions, rankings & KPI queries
│   ├── sql_analytics.py      # SQL runner & verified KPI JSON builder
│   ├── ml_pipeline.py        # Forecasting, trend classification & Isolation Forest
│   ├── genai_assistant.py    # Grounded Google Gemini API assistant with fallback
│   └── export_powerbi.py     # Star-schema CSV exports for Power BI Desktop
├── powerbi/
│   ├── datasets/             # Clean Power BI star-schema tables
│   └── HealthSight.pbix      # 4-page interactive Power BI dashboard
├── app/
│   └── streamlit_app.py      # Interactive 6-tab Streamlit application
├── models/                   # Serialized .joblib models & ml_evaluation_report.json
├── screenshots/              # Dashboard & application screenshots
├── requirements.txt
└── README.md
```

## Real-World Datasets & Methodology

1. **San Francisco Respiratory Virus Hospital Admissions (2024–2026)**
   - **Source:** San Francisco Department of Public Health (`data.sf.gov` Open Data API)
   - **Coverage:** 324 weekly observations across COVID-19, Influenza, and RSV (`2024-10-13` to `2026-11-07`; 288 observations from `2025-01-01` onward).
   - **Metrics:** Weekly `new_admissions`, `admission_rate_per_100k`, `activity_level`, and official `trend` status.

2. **NHS England Community Services Activity (July 2026)**
   - **Source:** NHS England (`digital.nhs.uk`, `csds-jul26-exp-core-data.zip`)
   - **Coverage:** 275,990 clean records covering July 2026 (`2026-07-01` to `2026-07-31`) across NHS providers, clinical service types, primary referral reasons, and age cohorts (`0–18`, `19–64`, `65+`).

3. **Methodological Separation:** Both sources are modeled in separate normalized tables linked via `data_ingestion_log` rather than joined across incompatible geographies or reporting frequencies.

## Verified Machine Learning Results (Chronological 80/20 Split)

- **Hospital Admission Forecasting:**
  - Ridge Regression: **MAE = 2.832**, **RMSE = 4.007**, **R² = 0.838**
  - Persistence Baseline: **MAE = 2.825**, **RMSE = 4.137**, **R² = 0.827**
  - Random Forest Regressor: **MAE = 3.017**, **RMSE = 4.643**, **R² = 0.782**
- **Disease Trend Classification (`Increasing` / `Decreasing` / `Stable`):**
  - Random Forest Classifier: **Accuracy = 0.667**, **Weighted F1 = 0.627**
  - Logistic Regression: **Accuracy = 0.635**, **Weighted F1 = 0.563**
- **Anomaly Detection (Isolation Forest):**
  - Flagged **7 anomalous weeks per virus** (21 total across 104 evaluated weeks).

## Dashboard & Application Screenshots

### Power BI 4-Page Interactive Dashboard (`powerbi/HealthSight.pbix`)

#### Page 1 – Healthcare Overview
![Power BI Healthcare Overview](screenshots/pbi_page1_overview.png)

#### Page 2 – Disease Intelligence
![Power BI Disease Intelligence](screenshots/pbi_page2_disease.png)

#### Page 3 – Predictive Analytics
![Power BI Predictive Analytics](screenshots/pbi_page3_predictive.png)

#### Page 4 – Hospital Operations (NHS England July 2026)
![Power BI Hospital Operations](screenshots/pbi_page4_operations.png)

---

### Streamlit Application & Gemini AI Assistant (`app/streamlit_app.py`)

#### Interactive Healthcare Overview & ML Analytics
![Streamlit Healthcare Overview](screenshots/streamlit_overview.png)

#### Gemini-Powered Healthcare Analytics Assistant
![Streamlit Gemini AI Assistant](screenshots/streamlit_gemini_ai.png)

## Setup & Execution

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key
```

Run the pipeline:
```bash
python src/etl.py
python src/sql_analytics.py
python src/ml_pipeline.py
python src/export_powerbi.py
streamlit run app/streamlit_app.py
```