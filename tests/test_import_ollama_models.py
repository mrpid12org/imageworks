import subprocess
import json
from pathlib import Path

REG_PATH = Path("configs/model_registry.json")


def _load():
    return json.loads(REG_PATH.read_text())


def test_import_ollama_models_dry_run_normalization():
    # Run dry-run; should not mutate registry
    before = REG_PATH.read_text()
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/import_ollama_models.py", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    after = REG_PATH.read_text()
    assert before == after, "Dry-run must not modify registry"
    out = proc.stdout.lower()
    # Expect slash-containing model normalized: hf.co/... path becomes hyphenated family
    # Normalization now replaces '/' with '-' so expect hyphenated form
    assert (
        "variant_family=hf.co-mradermacher-l3.1-dark-reasoning-lewdplay-evo-hermes-r1-uncensored-8b-i1-gguf"
        in out.replace("\n", " ")
    ), out
    # Ensure quant token preserved underscores (q6_k) not converted to dashes
    assert "quant=q6_k" in out, out


def test_import_ollama_models_real_insert(monkeypatch):
    # Monkeypatch subprocess to simulate minimal two-model list to keep test fast
    import scripts.import_ollama_models as imod

    class DummyProc:  # minimal object with attrs
        def __init__(self, stdout):
            self.stdout = stdout
            self.returncode = 0

    def fake_run(args, capture_output=True, text=True, check=False):  # noqa: D401
        if args[:3] == ["ollama", "list", "--format"]:
            # Provide JSON format list with two entries
            payload = json.dumps(
                [
                    {"name": "mini-test:latest", "size": 1024},
                    {"name": "testmodel:Q4_K", "size": 2048},
                ]
            )
            return DummyProc(payload)
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(imod.subprocess, "run", fake_run)
    # Call list_ollama_models to get fabricated list, then import
    models = imod.list_ollama_models()
    imod.import_models(
        models,
        backend="ollama",
        location="linux_wsl",
        dry_run=False,
        deprecate_placeholders=False,
    )
    reg = _load()
    names = {e["name"] for e in reg}
    # Expected variant names under Strategy A
    assert "mini-test-latest-ollama-gguf" in names
    assert "testmodel-ollama-gguf-q4_k" in names
