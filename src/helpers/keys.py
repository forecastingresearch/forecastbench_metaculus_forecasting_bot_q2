# Copyright (c) 2024, Forecasting Research Institute.
# Source: https://github.com/forecastingresearch/forecastbench
# License: Preserves original license terms. This repository is AGPL-3.0 overall.

"""utils for key-related tasks in llm-benchmark."""
import os
from google.cloud import secretmanager


def get_secret(secret_name, version_id="latest"):
    """
    Retrieve the payload of a specified secret version from Secret Manager.

    Accesses the Google Cloud Secret Manager to fetch the payload of a secret version
    identified by `project_id`, `secret_name`, and `version_id`. Decodes the payload
    from bytes to a UTF-8 string and returns it.
    """
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        raise ValueError("GCP PROJECT_ID environment variable not set.")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def get_secret_that_may_not_exist(secret_name, version_id="latest"):
    """Get a secret from Secret Manager but don't fail if it doesn't exist."""
    try:
        return get_secret(secret_name, version_id)
    except Exception:
        return None