"""Uvicorn entrypoint.

``python -m teuton_dashboard.main`` or ``teuton-dashboard`` after install.
"""
from __future__ import annotations

import logging

import uvicorn

from .settings import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    uvicorn.run(
        "teuton_dashboard.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
