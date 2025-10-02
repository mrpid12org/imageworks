Working Model Setups Registry
=============================

Purpose
-------
Curate reproducible, *known-working* end-to-end configurations for serving a vision-language model (vLLM or other backend) and running the Personal Tagger successfully. Each entry captures:

* Model identity (original repo + local path)
* Quantisation / format details
* Exact server launch command & critical flags
* Verified client (tagger) invocation with key overrides
* Date verified and commit hash (if available)
* Optional notes, caveats, and performance observations

Scope
-----
Only configurations that have produced successful multimodal (vision) outputs for all three tagger stages should be recorded as `status: success`. Failed attempts may optionally be documented with `status: failed` and a `failure_reason` to prevent duplicated effort.

File Layout
-----------
* `index.json` – Array of all setup metadata objects (lightweight manifest)
* `entries/<slug>.json` – Full detailed record for each working (or attempted) configuration

JSON Schema (informal)
----------------------
```
{
  "schema_version": "1.0",
  "model": {
    "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "local_path": "/abs/path/to/model",
    "served_model_name": "qwen2.5-vl-7b-awq",
    "format": "awq",
    "quantization": "awq_marlin",
    "vision": true
  },
  "server": {
    "backend": "vllm",
    "command": "nohup uv run vllm serve ...",
    "port": 24001,
    "flags": {"max-model-len": 4096, "enforce-eager": true},
    "chat_template": null
  },
  "tagger": {
    "command": "uv run imageworks-personal-tagger ...",
    "base_url": "http://localhost:24001/v1",
    "models": {
      "caption": "qwen2.5-vl-7b-awq",
      "keywords": "qwen2.5-vl-7b-awq",
      "description": "qwen2.5-vl-7b-awq"
    },
    "max_new_tokens": 256,
    "temperature": 0.2
  },
  "verification": {
    "date_utc": "2025-10-01T00:00:00Z",
    "git_commit": "<sha>",
    "images_processed": 22,
    "status": "success"
  },
  "notes": "Any special considerations, errors solved, etc.",
  "performance": {"avg_latency_s": null}
}
```

Conventions
-----------
* Use lowercase hyphenated slug for filename (e.g. `qwen2.5-vl-7b-awq_vllm.json`).
* Keep `command` exactly as executed (single line). Use placeholders only for private paths if required.
* Do **not** edit historical entries retroactively; add a new entry if configuration changes materially.

Workflow for Adding a New Setup
--------------------------------
1. Launch model server, confirm `/v1/models` lists served name.
2. Run personal tagger end-to-end on sample images.
3. Inspect outputs summary for populated caption, keywords, description.
4. Create entry JSON in `entries/`.
5. Append minimal manifest object to `index.json` (or regenerate tool in future).

Future Enhancements
-------------------
* Auto-capture environment (CUDA, vllm version) via helper script.
* Validation script to ensure all referenced paths exist.
* Optional benchmark harness capturing per-stage latency.
