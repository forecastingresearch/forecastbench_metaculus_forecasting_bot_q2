import asyncio
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from Bot.binary import get_binary_forecast
from src.helpers import keys, storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORGANIZATION_NAME = "Metaculus"
MODEL_NAME = "Panshul42 (Winner of Metaculus 2025Q2 AI Tournament)"
MODEL_ORGANIZATION = "Metaculus"

GCP_BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME", "")
FORECAST_FOLDER = os.environ.get("FORECAST_FOLDER", "")

SOURCES = {
    "market": ["infer", "manifold", "metaculus", "polymarket"],
    "dataset": ["acled", "dbnomics", "fred", "wikipedia", "yfinance"],
}


def load_secrets_into_environment():
    """Fetch secrets from GCP Secret Manager and load as environment variables."""
    logger.info("Loading secrets from GCP Secret Manager into environment...")

    secrets_to_load = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "serper": "GOOGLE_SERPER_API_KEY",
        "asknews-key": "ASKNEWS_SECRET",
        "asknews-key-id": "ASKNEWS_CLIENT_ID",
    }

    for secret_name, env_var in secrets_to_load.items():
        secret_value = keys.get_secret_that_may_not_exist(secret_name)
        if secret_value:
            os.environ[env_var] = secret_value
            logger.info(
                f"Loaded secret '{secret_name}' into env var '{env_var}'."
            )
        else:
            logger.warning(
                f"Secret '{secret_name}' not found; the bot may fail if it needs this key."
            )

def make_writer(meta: Dict[str, Any]):
    prefix = f"[FB question] set={meta.get('question_set')} source={meta.get('source')} id={meta.get('id')}"
    if meta.get("resolution_date"):
        prefix += f" res_date={meta['resolution_date']}"
    def _w(msg: str):
        print(f"{prefix} :: {msg}")
    return _w


def build_question_details(q: pd.Series, question_set_name: str, resolution_date: Optional[str] = None) -> Dict[str, Any]:
    d = {
        "title": q.get("question"),
        "resolution_criteria": q.get("resolution_criteria"),
        "description": q.get("background"),
        "fine_print": q.get("fine_print", "N/A"),
        "id": q.get("id"),
        "source": q.get("source"),
        "question_set": question_set_name,
        "is_dataset": q.get("source") not in ["infer", "manifold", "metaculus", "polymarket"],
    }
    if resolution_date is not None:
        d["resolution_date"] = resolution_date
    if d["is_dataset"]:
        if "freeze_datetime_value" in q:
            d["freeze_datetime_value"] = q["freeze_datetime_value"]
        if "freeze_datetime_value_explanation" in q:
            d["freeze_datetime_value_explanation"] = q["freeze_datetime_value_explanation"]
    return d

def get_prediction_from_bot(question_details: Dict[str, Any], writer) -> Optional[float]:
    try:
        final_prob, _ = asyncio.run(get_binary_forecast(question_details, write=writer))
        if final_prob is None:
            writer("Bot returned None. Skipping.")
            return None
        return float(final_prob)
    except Exception as e:
        writer(f"Error running bot: {e}. Skipping.")
        return None


def download_question_set(forecast_due_date: str) -> Dict[str, Any]:
    """Download latest question set from ForecastBench GitHub."""
    filename = f"{forecast_due_date}-llm.json"
    url = (
        "https://github.com/forecastingresearch/forecastbench-datasets/raw/main/"
        f"datasets/question_sets/{filename}"
    )
    local_filename = f"/tmp/{filename}"

    logger.info(f"Downloading question set from {url}...")
    try:
        urllib.request.urlretrieve(url, local_filename)
        with open(local_filename, "r", encoding="utf-8") as f:
            question_set = json.load(f)
        logger.info("Successfully downloaded and loaded question set.")
        return question_set
    except Exception as e:
        logger.error(f"Failed to download or read question set. Error: {e}")
        sys.exit(1)


def create_all_forecasts(questions_df: pd.DataFrame, question_set_name: str) -> List[Dict[str, Any]]:
    """
    Generate forecasts for all (standard) questions.

    - We have already filtered out combination questions upstream.
    - If a prediction fails (None), we DO NOT emit a forecast row.
      ForecastBench will impute 0.5 for the missing forecast.
    """
    all_forecasts: List[Dict[str, Any]] = []

    market_mask = questions_df["source"].isin(SOURCES["market"])
    dataset_mask = ~market_mask

    skipped = 0
    emitted = 0

    for index, q in questions_df.iterrows():
        is_market = bool(market_mask.iloc[index])
        is_dataset = bool(dataset_mask.iloc[index])

        if is_market:
            details = build_question_details(q, question_set_name)
            writer = make_writer(details)
            prediction = get_prediction_from_bot(details, writer)
            if prediction is None:
                skipped += 1
                continue

            all_forecasts.append({
                "id": q["id"],
                "source": q["source"],
                "forecast": prediction,
                "resolution_date": None,
                "reasoning": None,
                "direction": None,
            })
            emitted += 1

        elif is_dataset:
            # For dataset questions, FB expects forecasts at each resolution date.
            for res_date in q.get("resolution_dates", []):
                details = build_question_details(q, question_set_name, resolution_date=res_date)
                writer = make_writer(details)
                prediction = get_prediction_from_bot(details, writer)
                if prediction is None:
                    skipped += 1
                    continue

                all_forecasts.append({
                    "id": q["id"],
                    "source": q["source"],
                    "forecast": prediction,
                    "resolution_date": res_date,
                    "reasoning": None,
                    "direction": None,
                })
                emitted += 1

    logger.info(f"Forecasts emitted: {emitted}. Skipped (failed): {skipped}.")
    return all_forecasts


def driver(_: Any):
    """Main entry point for GCP Cloud Run Job."""
    load_secrets_into_environment()

    logger.info("Starting ForecastBench submission job...")
    forecast_due_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # 1) Download and parse the question set
    question_set_data = download_question_set(forecast_due_date)
    question_set_name = question_set_data["question_set"]
    questions_df = pd.DataFrame(question_set_data["questions"])

    # 2) TEMP TESTING STEP: Remove combination rows.
    # Combination questions show up as non-string IDs; keep only "standard" rows.
    before = len(questions_df)
    questions_df = questions_df[questions_df["id"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)
    after = len(questions_df)
    logger.info(f"Filtered out combination rows for testing: {before - after} removed; {after} standard remain.")

    # 3) Generate forecasts
    logger.info(f"Generating forecasts for {len(questions_df)} questions...")
    forecasts_list = create_all_forecasts(questions_df, question_set_name)
    logger.info(f"Successfully generated {len(forecasts_list)} forecast rows.")

    # 4) Compile the submission payload
    submission_data = {
        "organization": ORGANIZATION_NAME,
        "model": MODEL_NAME,
        "model_organization": MODEL_ORGANIZATION,
        "question_set": question_set_name,
        "forecasts": forecasts_list,
    }

    submission_filename = f"{forecast_due_date}.{ORGANIZATION_NAME}.Metaculus.Panshul42.json"
    local_filepath = f"/tmp/{submission_filename}"
    with open(local_filepath, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, indent=4)
    logger.info(f"Submission file created at {local_filepath}")

    # 5) Upload to GCS
    if not GCP_BUCKET_NAME:
        logger.error("GCP_BUCKET_NAME is not set; cannot upload submission.")
        sys.exit(1)

    logger.info(f"Uploading to gs://{GCP_BUCKET_NAME}/{FORECAST_FOLDER}...")
    try:
        storage.upload(
            bucket_name=GCP_BUCKET_NAME,
            local_filename=local_filepath,
            destination_folder=FORECAST_FOLDER,
        )
        logger.info("Successfully uploaded forecasts to GCP.")
    except Exception as e:
        logger.error(f"Failed to upload to GCP. Error: {e}")
        sys.exit(1)

    logger.info("ForecastBench submission job finished successfully! 🎉")


if __name__ == "__main__":
    driver(None)
