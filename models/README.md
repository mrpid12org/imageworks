# Model Directory Stub

Models are intentionally **kept outside the repository**. Use the shared
`$HOME/ai-models/weights` directory referenced in `docs/dev-env/ide-setup-wsl-vscode.md`.
The `IMAGEWORKS_MODEL_ROOT` environment variable (set in `.bashrc`) should point
at that location so scripts and CLIs can discover the weights.

Typical layout:

```
~/ai-models/weights/
├── Qwen2.5-VL-7B-Instruct-AWQ/     # default LMDeploy backend
├── Qwen2-VL-2B-Instruct/           # vLLM fallback
└── Qwen2-VL-7B-Instruct/           # full-precision reference
```

This repository keeps only this README so the directory exists for tools that
expect it, but **no weights should live under `imageworks/models/`**.
