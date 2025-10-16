import json
from pathlib import Path


def _load(registry_path: Path):
    return json.loads(registry_path.read_text(encoding="utf-8"))


def test_import_ollama_models_dry_run_normalization(isolated_configs_dir, capsys):
    import scripts.import_ollama_models as imod

    # Use bundled fallback sample to avoid shelling out to real ollama
    models = [dict(item) for item in imod._FALLBACK_SAMPLE_MODELS]
    reg_path = isolated_configs_dir / "model_registry.json"
    before = reg_path.read_text(encoding="utf-8")
    count = imod.import_models(
        models,
        backend="ollama",
        location="linux_wsl",
        dry_run=True,
        deprecate_placeholders=False,
        purge=False,
    )
    assert count == len(models)
    captured = capsys.readouterr().out.lower()
    assert (
        "variant_family=hf.co-mradermacher-l3.1-dark-reasoning-lewdplay-evo-hermes-r1-uncensored-8b-i1-gguf"
        in captured.replace("\n", " ")
    )
    assert "quant=q6_k" in captured
    # Dry run must not mutate registry copy
    after = reg_path.read_text(encoding="utf-8")
    assert before == after


def test_import_ollama_models_real_insert(monkeypatch, isolated_configs_dir):
    import scripts.import_ollama_models as imod

    class DummyProc:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.returncode = 0

    def fake_run(args, capture_output=True, text=True, check=False):  # noqa: D401
        if args[:3] == ["ollama", "list", "--format"]:
            payload = json.dumps(
                [
                    {"name": "mini-test:latest", "size": 1024},
                    {"name": "testmodel:Q4_K", "size": 2048},
                ]
            )
            return DummyProc(payload)
        raise FileNotFoundError("ollama binary not available")

    monkeypatch.setattr(imod.subprocess, "run", fake_run)
    models = imod.list_ollama_models()
    imported = imod.import_models(
        models,
        backend="ollama",
        location="linux_wsl",
        dry_run=False,
        deprecate_placeholders=False,
        purge=False,
    )
    assert imported == 2
    reg = _load(isolated_configs_dir / "model_registry.json")
    names = {e["name"] for e in reg}
    assert "mini-test-latest" in names
    assert "testmodel_(Q4_K)" in names
