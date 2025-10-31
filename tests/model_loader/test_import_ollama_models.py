import pytest

from scripts import import_ollama_models as importer


class StubClient:
    def __init__(self):
        self.list_called = False
        self.show_calls = []

    def list_models(self):
        self.list_called = True
        return [
            {
                "name": "qwen2.5vl:7b",
                "size": 6_000_000_000,
                "details": {"quantization_level": "q6_k"},
            }
        ]

    def show_model(self, name: str):
        self.show_calls.append(name)
        return {
            "details": {
                "architecture": "llama",
                "parameters": "7B",
                "quantization_level": "q6_k",
            }
        }

    def close(self):
        pass


def test_run_import_dry_run_uses_http_client(monkeypatch, capsys):
    client = StubClient()
    count = importer.run_import(
        client=client,
        backend="ollama",
        location="test",
        dry_run=True,
        deprecate_placeholders=False,
        purge_existing=False,
    )
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert count == 1
    assert client.list_called
    assert client.show_calls == ["qwen2.5vl:7b"]


def test_run_import_dry_run_fallback(monkeypatch, capsys):
    class FailingClient:
        def list_models(self):
            raise importer.OllamaError("boom")

        def close(self):
            pass

    monkeypatch.setattr(importer, "OllamaClient", lambda: FailingClient())
    count = importer.run_import(dry_run=True)
    captured = capsys.readouterr()
    assert "sample dataset" in captured.err
    assert count == len(importer._FALLBACK_SAMPLE_MODELS)


def test_run_import_errors_when_not_dry(monkeypatch):
    class FailingClient:
        def list_models(self):
            raise importer.OllamaError("boom")

        def close(self):
            pass

    monkeypatch.setattr(importer, "OllamaClient", lambda: FailingClient())
    with pytest.raises(importer.OllamaError):
        importer.run_import(dry_run=False)


def test_family_and_quant_derivation_hf_sources():
    family, quant = importer._derive_family_and_quant(
        "hf.co/huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2:latest"
    )
    assert family == "huihui-gpt-oss-20b-abliterated-v2-latest"
    assert quant == "mxfp4"


def test_family_and_quant_derivation_pixtral():
    family, quant = importer._derive_family_and_quant(
        "hf.co/Hyphonical/Pixtral-12B-Captioner-Relaxed-Q4_K_M-GGUF:Q4_K_M"
    )
    assert family == "pixtral-12b-captioner-relaxed"
    assert quant == "q4_k_m"
