import os
import json
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import GEMINI_API_KEY, GEMINI_MODEL, PROCESSED_DIR, MODELS_DIR

KPI_JSON_PATH = os.path.join(PROCESSED_DIR, "kpi_summary.json")
ML_REPORT_PATH = os.path.join(MODELS_DIR, "ml_evaluation_report.json")

SYSTEM_INSTRUCTION = """
You are the HealthSight Healthcare Analytics AI Assistant.
Your role is to help hospital operations analysts and healthcare data teams interpret verified statistics, machine learning forecasts, disease trends, and anomaly alerts.

STRICT RULES:
1. GROUNDING ONLY: Base every number, date, percentage, model metric, and provider name strictly on the VERIFIED ANALYTICS CONTEXT provided in the prompt. Never invent, extrapolate, or guess unprovided statistics.
2. SOURCE SEPARATION: Never merge or confuse San Francisco weekly respiratory hospital admissions (COVID-19, Influenza, RSV) with NHS England July 2026 community services activity. Always specify which dataset and timeframe a metric comes from.
3. OPERATIONAL FRAMING (NOT MEDICAL ADVICE): Frame all staffing, bed-capacity, triage, or supply-chain ideas strictly as "Data-Driven Operational Suggestions (Not Medical Advice)". Do not provide clinical diagnoses, treatment protocols, or medical advice.
4. CLARITY & STRUCTURE: Use concise bullet points, bold key figures, and clear headings when answering.
"""


def load_verified_context() -> dict:
    """Load verified SQL KPIs and ML evaluation outputs from disk."""
    context = {}
    if os.path.exists(KPI_JSON_PATH):
        with open(KPI_JSON_PATH, "r", encoding="utf-8") as f:
            context["sql_kpis"] = json.load(f)
    if os.path.exists(ML_REPORT_PATH):
        with open(ML_REPORT_PATH, "r", encoding="utf-8") as f:
            context["ml_evaluation_and_forecasts"] = json.load(f)
    return context


def ask_healthsight_assistant(user_question: str, api_key: str | None = None) -> str:
    """Query Google Gemini using verified HealthSight analytics context with error handling."""
    effective_key = (api_key or GEMINI_API_KEY or "").strip()
    if not effective_key or effective_key == "your_actual_google_ai_studio_api_key_here":
        return (
            "⚠️ **Gemini API Key Missing:** Please add your `GEMINI_API_KEY` to the `.env` file "
            "(or enter it in the Streamlit sidebar) to enable AI-powered operational insights."
        )

    verified_data = load_verified_context()
    if not verified_data:
        return "⚠️ **Analytics Context Missing:** Please run `src/sql_analytics.py` and `src/ml_pipeline.py` first."

    prompt = (
        "### VERIFIED ANALYTICS CONTEXT (JSON)\n"
        f"{json.dumps(verified_data, indent=2)}\n\n"
        "### USER QUESTION\n"
        f"{user_question}\n\n"
        "Provide a clear, accurate response grounded strictly in the verified JSON context above. "
        "End with a short disclaimer reminding the user that operational insights are data-driven suggestions and not medical advice."
    )

    try:
        client = genai.Client(
            api_key=effective_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return response.text or "No response text returned by Gemini."

    except APIError as e:
        if getattr(e, "code", None) == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return (
                "⚠️ **Gemini Free-Tier Quota Limit Reached (HTTP 429):** "
                "Too many requests were sent in a short window. Please wait 30–60 seconds and try again."
            )
        return f"⚠️ **Gemini API Error:** {e}"
    except Exception as e:
        return f"⚠️ **Unexpected Assistant Error:** {e}"


if __name__ == "__main__":
    print("=" * 72)
    print("TESTING HEALTHSIGHT GEMINI HEALTHCARE ANALYTICS ASSISTANT")
    print("=" * 72)

    sample_q = (
        "Summarise the latest San Francisco respiratory admission trends and next-week forecasts, "
        "highlight any recent anomalies, and provide 3 operational resource-planning suggestions."
    )
    print(f"Prompt: {sample_q}\n")
    reply = ask_healthsight_assistant(sample_q)
    print(reply)