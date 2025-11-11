#!/usr/bin/env python3
"""Download deterministic IQA model weights (NIMA + MUSIQ) into the shared model store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import time

import requests

from imageworks.tools.model_downloader.config import get_config


NIMA_WEIGHTS = {
    "aesthetic": "https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5",
    "technical": "https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_technical_0.11.hdf5",
}

MUSIQ_HANDLES = {
    "spaq": "https://tfhub.dev/google/musiq/spaq/1",
}


def _model_root() -> Path:
    config = get_config()
    root = config.linux_wsl.root / "weights"
    root.mkdir(parents=True, exist_ok=True)
    return root / "judge-iqa" / "nima"


def _download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"✔ {path} already present; skipping.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {url} → {path} ...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(path)
    print(f"✅ Saved {path}")


def _musiq_cache_dir() -> Path:
    config = get_config()
    cache = config.linux_wsl.root / "weights" / "judge-iqa" / "musiq" / "tfhub-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_musiq_cache(force: bool) -> Path:
    cache_dir = _musiq_cache_dir()
    if not force and any(cache_dir.iterdir()):
        print(f"✔ MUSIQ cache already populated at {cache_dir}")
        return cache_dir

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TFHUB_CACHE"] = str(cache_dir)
    os.environ["TFHUB_CACHE_DIR"] = str(cache_dir)
    import tensorflow as tf  # noqa: F401  (ensures TF is initialised before TF-Hub)
    import tensorflow_hub as hub

    def _clear_directory(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    print(f"⬇️  Downloading MUSIQ TF-Hub modules into {cache_dir} ...")
    for name, handle in MUSIQ_HANDLES.items():
        print(f"   • Resolving {name}: {handle}")
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                hub.load(handle)
                break
            except Exception as exc:  # noqa: BLE001
                print(
                    f"      ⚠️  Failed to download {name} (attempt {attempt}/{retries}): {exc}"
                )
                _clear_directory(cache_dir)
                tmp_cache = Path(tempfile.gettempdir()) / "tfhub_modules"
                if tmp_cache.exists():
                    shutil.rmtree(tmp_cache, ignore_errors=True)
                if attempt == retries:
                    raise
                time.sleep(2.0)

    if not any(cache_dir.iterdir()):
        tmp_cache = Path(tempfile.gettempdir()) / "tfhub_modules"
        if tmp_cache.exists() and any(tmp_cache.iterdir()):
            print(
                f"⚠️  TensorFlow Hub cached modules under {tmp_cache}; copying into {cache_dir}"
            )
            for item in tmp_cache.iterdir():
                dest = cache_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

    return cache_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload weights even if files already exist.",
    )
    args = parser.parse_args()

    nima_dir = _model_root()
    nima_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using NIMA directory: {nima_dir}")

    for flavor, url in NIMA_WEIGHTS.items():
        filename = Path(url).name
        _download(url, nima_dir / filename, force=args.force)

    musiq_cache = _download_musiq_cache(force=args.force)
    print(f"MUSIQ cache ready at: {musiq_cache}")

    print(
        "All IQA weights downloaded. Judge Vision can now load IQA models deterministically."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
