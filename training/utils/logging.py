from __future__ import annotations

import logging


def silence_external_info_logs() -> None:
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
