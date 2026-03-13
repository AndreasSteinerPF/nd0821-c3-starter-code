from __future__ import annotations

import argparse
import json
import os
import sys

import requests

from starter.train_model import SAMPLE_REQUESTS_PATH, ensure_artifacts

DEFAULT_URL = os.environ.get("LIVE_API_URL", "http://127.0.0.1:8000/predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Prediction endpoint URL.")
    parser.add_argument(
        "--label",
        choices=["<=50K", ">50K"],
        default=">50K",
        help="Sample request payload to send.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_artifacts()
    args = parse_args()
    sample_requests = json.loads(SAMPLE_REQUESTS_PATH.read_text(encoding="utf-8"))
    try:
        response = requests.post(
            args.url,
            json=sample_requests[args.label],
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Status: {response.status_code}")
    print(response.text)

    if not response.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
