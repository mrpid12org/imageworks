from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, Iterable, Optional

import json
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from imageworks.libs.vision.mono import check_monochrome


app = FastAPI(title="Imageworks - Mono Checker API", version="0.1.0")


def _iter_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts = {e.strip().lstrip(".").lower() for e in exts if e.strip()}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lstrip(".").lower() in exts:
            yield p


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/mono/check")
async def mono_check(
    image: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Body(default=None),
    neutral_tol: int = Body(default=2),
    toned_pass: float = Body(default=6.0),
    toned_query: float = Body(default=10.0),
):
    img_path: Optional[str] = None
    tmp: Optional[NamedTemporaryFile] = None
    try:
        if path:
            img_path = path
        elif image is not None:
            tmp = NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(await image.read())
            tmp.flush()
            img_path = tmp.name
        else:
            raise HTTPException(
                status_code=400, detail="Provide 'path' or upload 'image'."
            )
        res = check_monochrome(img_path, neutral_tol, toned_pass, toned_query)
        return JSONResponse(asdict(res))
    finally:
        if tmp is not None:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/mono/batch")
def mono_batch(
    folder: str = Body(...),
    exts: str = Body("jpg,jpeg"),
    neutral_tol: int = Body(default=2),
    toned_pass: float = Body(default=6.0),
    toned_query: float = Body(default=10.0),
):
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    def _gen() -> Generator[bytes, None, None]:
        for p in _iter_files(root, [e for e in exts.split(",") if e.strip()]):
            res = check_monochrome(str(p), neutral_tol, toned_pass, toned_query)
            obj = asdict(res)
            obj["path"] = str(p)
            yield (json.dumps(obj) + "\n").encode("utf-8")

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run(
        "imageworks.apps.mono_checker.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )
