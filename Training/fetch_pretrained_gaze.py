#!/usr/bin/env python3
"""fetch the Apache-2.0 OpenVINO gaze initializer with checksum verification"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


ARCHIVE_URL = (
    "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/"
    "091_gaze-estimation-adas-0002/resources_new_onnx_only.tar.gz"
)
ARCHIVE_SHA256 = "eecddc6655ac1709ab3ce5dfcc9373e2b7f1b58712ef892be78f33e72e5831e1"
MODEL_NAME = "gaze_estimation_adas_0002.onnx"
MODEL_SHA256 = "f8f13707401547c7c5e146ba22da1bd6ada00f47ebf347be6d97c5acfaf4c2bd"
DOWNLOAD_ATTEMPTS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_NAME
    if model_path.is_file() and sha256(model_path) == MODEL_SHA256:
        print(model_path)
        return

    with tempfile.TemporaryDirectory(prefix="aspectus-pretrained-") as directory:
        archive = Path(directory) / "model.tar.gz"
        for attempt in range(DOWNLOAD_ATTEMPTS):
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
            if sha256(archive) == ARCHIVE_SHA256:
                break
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise ValueError("pretrained archive checksum mismatch")
            time.sleep(1)
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(MODEL_NAME)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"{MODEL_NAME} is missing from the pretrained archive")
            temporary_model = Path(directory) / MODEL_NAME
            temporary_model.write_bytes(extracted.read())
        if sha256(temporary_model) != MODEL_SHA256:
            raise ValueError("pretrained model checksum mismatch")
        temporary_model.replace(model_path)
    print(model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    fetch(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
