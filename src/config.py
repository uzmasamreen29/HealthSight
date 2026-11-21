import os
from dotenv import load_dotenv

load_dotenv()

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
NHS_RAW_DIR = os.path.join(RAW_DIR, "nhs")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Database Path
DB_PATH = os.path.join(PROCESSED_DIR, "healthsight.db")

# Dataset 1: San Francisco Respiratory Virus Hospital Admissions (Live Official CSV Endpoint)
SF_RESPIRATORY_URL = "https://data.sf.gov/api/v3/views/ppwr-akuv/export.csv?accessType=DOWNLOAD"
SF_RAW_CSV = os.path.join(RAW_DIR, "sf_respiratory_raw.csv")

# Dataset 2: NHS England Community Services Data Set (July 2026 Official Release)
NHS_ZIP_PATH = os.path.join(NHS_RAW_DIR, "csds-jul26-exp-core-data.zip")

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Ensure required directories exist
for directory in [RAW_DIR, NHS_RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)