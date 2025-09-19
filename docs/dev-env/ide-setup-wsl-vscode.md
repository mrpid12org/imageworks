# Reproducible Dev Environment: Windows + WSL + VS Code

This guide captures the environment we standardise on for Imageworks and related ML/photo projects. It explains what to install, why each choice matters, and the common pitfalls we already hit so you can avoid them.

## Overview
- Develop inside **WSL2 (Ubuntu)** for predictable Linux tooling while keeping the Windows UI.
- Use **VS Code Remote – WSL** so editor extensions share the same interpreter as the code.
- Manage Python with **uv** for fast, reproducible environments and lockfiles.
- Keep heavyweight model artefacts in shared caches instead of inside repositories.
- Automate formatting, linting, and testing with `pre-commit` and `pytest`.

## 0. Install & Verify WSL2
### Why WSL2
- Linux is the native platform for CUDA wheels, FAISS, and most ML packages.
- WSL isolates the Linux userland, giving repeatable builds across projects.
- You still retain Windows UX, GPU access, and filesystem integration.

### Install Steps (one time)
```powershell
wsl --install -d Ubuntu
```
Reboot if prompted, then create your Linux user (e.g. `stewa`).

### GPU Driver Check
1. Install the latest NVIDIA Studio/Game Ready driver on Windows.
2. In WSL run:
```bash
nvidia-smi
```
Confirm the GPU is visible before installing CUDA-enabled Python wheels.

## 1. Use VS Code Remote – WSL
- Install VS Code for Windows and add the **Remote – WSL**, **Python**, and **Docker** extensions (optionally GitHub PRs, GitLens, Copilot).
- Open a WSL window (`F1 → WSL: Connect to WSL`) and then open the repository from `/home/<user>/code/...`.
- All terminals, Python processes, and extensions now run inside Ubuntu.

## 2. Directory Conventions
Keep repositories and model caches organised to avoid ad-hoc paths later.
```bash
mkdir -p ~/code
mkdir -p ~/ai-models/huggingface
mkdir -p ~/ai-models/weights
```
Set the Hugging Face caches so every project reuses downloaded models:
```bash
echo 'export HF_HOME=$HOME/ai-models/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=$HF_HOME/hub' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=$HF_HOME' >> ~/.bashrc
source ~/.bashrc
rm -rf ~/.cache/huggingface
ln -s "$HF_HOME" ~/.cache/huggingface
```
A symlink keeps straggler tools (that ignore environment variables) writing into the shared cache.

## 3. Python Toolchain: uv
Why uv?
- Single binary that handles environments **and** resolver (pip + virtualenv + pip-tools in one).
- Produces a lockfile so collaborators get identical dependencies.
- Plays nicely with system Python—no conda base environment to babysit.

Install once:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```
Keep Conda for legacy projects if needed, but disable auto-activation of `base`.

## 4. Git and SSH Setup
Configure Git defaults (edit to match your details):
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global core.autocrlf input
git config --global pull.rebase false
```
Create an SSH key and register it with GitHub:
```bash
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub  # paste into GitHub → Settings → SSH keys
ssh -T git@github.com      # expect the “successfully authenticated” message
```
**Tip:** Use `git mv` only after a file is tracked. For untracked files move them with `mv` and then `git add` (history is not preserved, but it avoids the silent `git mv` no-op).

## 5. Reusable Project Scaffold
The pattern we reuse for new repos:
```bash
cd ~/code
mkdir myproject && cd myproject
git init -b main
```
Add a `.gitignore` that keeps venvs, caches, models, and generated reports out of Git:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.vscode/
*.egg-info
src/*.egg-info
ai-models/
*.pt
*.bin
*.ckpt
*.safetensors
*.jsonl
*.csv
```
Create the `src/` layout so imports always use the installed package:
```bash
mkdir -p src/myproject
printf '' > src/myproject/__init__.py
mkdir -p tests
```

## 6. pyproject.toml Template
Start with a single source of truth for metadata, dependencies, and CLIs:
```toml
[project]
name = "myproject"
version = "0.1.0"
description = "Short description"
readme = "README.md"
authors = [{ name = "Your Name", email = "you@example.com" }]
requires-python = ">=3.12"
dependencies = [
    "typer>=0.17.4",
    "rich>=14.1.0",
    "pydantic>=2.11.9",
    "fastapi>=0.116.2",
    "uvicorn[standard]>=0.35.0"
    # add ML deps as needed: "torch>=2.8.0", "opencv-python-headless>=4.12", "faiss-cpu>=1.12.0", ...
]

[project.scripts]
# example CLI
# myproject-cli = "myproject.apps.demo.cli:app"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]

[dependency-groups]
dev = [
    "black>=25.1.0",
    "ruff>=0.13.0",
    "pytest>=8.4.2",
    "pre-commit>=4.3.0",
    "huggingface-hub[cli]>=0.35.0"
]
```
Setuptools keeps compatibility high while we iterate on pure-Python packages.

## 7. Pre-commit Hooks
Stop formatting drift and lint issues at commit time:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks: [{ id: black }]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{ id: ruff }]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```
Install and run:
```bash
uv sync
uvx pre-commit install
uvx pre-commit run --all-files
```
If Black rewrites files the hook will fail intentionally—`git add` the changes and retry the commit.

## 8. pytest Configuration
Silence known noisy warnings so failures stand out:
```ini
[pytest]
addopts = -q
filterwarnings =
    ignore:.*SwigPy.*:DeprecationWarning
    ignore:.*mode.*deprecated.*Pillow 13.*:DeprecationWarning
```
Run tests via:
```bash
uv run pytest -q
```
Running once immediately after bootstrapping gives you a known-good baseline.

## 9. VS Code Settings
Optional workspace settings that keep the editor aligned with the project env:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "[python]": { "editor.defaultFormatter": "ms-python.black-formatter" },
  "ruff.lint.args": ["--fix"]
}
```
Explicit interpreter selection avoids the common “works in terminal but not VS Code” trap.

## 10. Packaging & CLI Example
Typer makes it easy to expose commands:
```python
import typer

app = typer.Typer()

@app.command()
def hello(name: str = "world"):
    print(f"Hello, {name}!")
```
Register it under `[project.scripts]`, run `uv sync`, then:
```bash
uv run myproject-cli hello --name Stewart
```
Console scripts guarantee you import the package as installed, not from the working directory.

## 11. GPU & Cache Sanity Checks
Verify the environment before committing to long runs:
```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
PY

echo $HF_HOME
echo $TRANSFORMERS_CACHE
readlink -f ~/.cache/huggingface
```
If CUDA is missing here it will be missing for your app and tests.

## 12. Git Do & Don’t
**Do**
- Initialise Git (`git init -b main`) and `.gitignore` before generating artefacts.
- Commit small, coherent changes with clear messages.
- Keep model weights, caches, and run outputs out of Git.

**Don’t**
- Work in a non-WSL VS Code window while running code inside WSL—paths and interpreters will diverge.
- Mix HTTPS and SSH remotes on the same repo.
- Rely on implicit imports from the working directory; use the `src/` layout and install the package.

## 13. When to Reach for Docker
WSL covers day-to-day development. Use Docker only when you need to:
- Pin system libraries/GLIBC versions for deployment.
- Ship a service to CI/production that must mirror a container image.
If you do, install NVIDIA Container Toolkit, mount your repo plus `~/ai-models`, and run the same pytest/pre-commit workflow inside the container.

## 14. New Project Checklist
1. Open a WSL VS Code window.
2. `git init -b main` and create `.gitignore`.
3. Create the `src/` layout with an empty `__init__.py`.
4. Write `pyproject.toml` (metadata + deps + CLIs).
5. `uv sync` to create the environment.
6. Add `.pre-commit-config.yaml` and install hooks.
7. `uvx pre-commit run --all-files` to normalise the repo.
8. Add `pytest.ini` and run `uv run pytest -q` for a baseline.
9. Push to GitHub once the baseline is green.

## 15. Troubleshooting Patterns
- **CLI cannot import your package** → ensure `[tool.setuptools]` and `[tool.setuptools.packages.find]` entries are present, then `uv sync`.
- **Models download to unexpected locations** → check `$HF_HOME`, `$TRANSFORMERS_CACHE`, and the symlink at `~/.cache/huggingface`.
- **Black “failed” during pre-commit** → it reformatted files; stage them and retry.
- **CUDA wheels installed but `torch.cuda.is_available()` is `False`** → update Windows GPU drivers so WSL exposes the device.

## 16. Next Steps
Keep this guide in `docs/dev-env/`. For future automation we can wrap the scaffold in a Cookiecutter template so new repositories spin up with the agreed defaults in under a minute.
