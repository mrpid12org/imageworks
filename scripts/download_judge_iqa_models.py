#!/usr/bin/env python3
"""Download deterministic IQA model weights (NIMA + MUSIQ) into the shared model store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import tarfile
import tempfile
from datetime import datetime
import hashlib

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

    print(f"⬇️  Downloading MUSIQ TF-Hub modules into {cache_dir} ...")
    for name, handle in MUSIQ_HANDLES.items():
        print(f"   • Resolving {name}: {handle}")
        archive_url = f"{handle}?tf-hub-format=compressed"
        response = requests.get(archive_url, stream=True, timeout=120)
        response.raise_for_status()

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
        tmp_path = Path(tmp_file.name)
        try:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    tmp_file.write(chunk)
            tmp_file.flush()
        finally:
            tmp_file.close()

        try:
            module_hash = hashlib.sha1(handle.encode("utf-8")).hexdigest()
            target_dir = cache_dir / module_hash
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tmp_path, "r:gz") as archive:
                archive.extractall(path=target_dir)

            descriptor = cache_dir / f"{module_hash}.descriptor.txt"
            descriptor.write_text(
                "\n".join(
                    [
                        f"Module: {handle}",
                        f"Download Time: {datetime.now().isoformat()}",
                        f"Downloader Hostname: {socket.gethostname()} (PID:{os.getpid()})",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"      ✅ Cached MUSIQ {name} module at {target_dir}")
        finally:
            tmp_path.unlink(missing_ok=True)

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
