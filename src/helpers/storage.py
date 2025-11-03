# Copyright (c) 2024, Forecasting Research Institute.
# Source: https://github.com/forecastingresearch/forecastbench
# License: Preserves original license terms. This repository is AGPL-3.0 overall.

"""Simplify Cloud Storage interactions."""

import os

from . import storage_upload_file

def upload(
    bucket_name: str,
    local_filename: str,
    destination_folder: str = "",
    *,
    filename: str = None,
):
    """Facilitate uploading file to cloud storage."""
    if not filename:
        filename = os.path.basename(local_filename)
    destination_filename = f"{destination_folder}/{filename}" if destination_folder else filename

    storage_upload_file.upload_blob(
        bucket_name,
        local_filename,
        destination_filename,
    )