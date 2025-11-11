import json
from pathlib import Path

import pytest

from imageworks.tools.model_downloader import architecture as arch_module
from imageworks.tools.model_downloader.architecture import (
    ArchitectureResult,
    collect_architecture_metadata,
    infer_default_vllm_arguments,
    merge_architecture_metadata,
)
from imageworks.model_loader.runtime_metadata import merge_runtime_payload


def test_collect_architecture_from_config(tmp_path: Path) -> None:
    config = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "rope_theta": 10000.0,
        "kv_cache_dtype": "fp16",
        "vision_config": {
            "image_size": 448,
            "patch_size": 14,
            "vision_hidden_size": 1024,
            "vision_num_hidden_layers": 24,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = collect_architecture_metadata(tmp_path)

    assert result.fields["num_layers"] == 28
    assert result.fields["num_attention_heads"] == 16
    assert result.fields["num_kv_heads"] == 8
    assert result.fields["hidden_size"] == 4096
    assert result.fields["intermediate_size"] == 11008
    assert result.fields["context_length"] == 8192
    assert result.fields["rope_theta"] == 10000.0
    assert result.fields["kv_precision"] == "fp16"
    assert result.fields["vision"]["image_size"] == 448
    assert "config:" in result.sources[0]


def test_merge_architecture_metadata_overwrites(tmp_path: Path) -> None:
    base = {"num_layers": 16, "sources": ["manual"]}
    incoming = ArchitectureResult(fields={"num_layers": 24}, sources=["config"])
    merged = merge_architecture_metadata(base, incoming)

    assert merged["num_layers"] == 24
    assert "manual" in merged["sources"]
    assert "config" in merged["sources"]
    assert "collected_at" in merged


def test_infer_default_vllm_arguments() -> None:
    result = ArchitectureResult(fields={"context_length": 8192})
    args = infer_default_vllm_arguments(result)
    assert "--max-model-len" in args
    idx = args.index("--max-model-len")
    assert int(args[idx + 1]) == int(8192 * 0.6)
    assert "--max-num-seqs" in args
    assert "--gpu-memory-utilization" in args


def test_merge_runtime_payload() -> None:
    arch_meta = {"sources": ["config"]}
    runtime_payload = {
        "metrics": {"runtime_context_tokens": 4096},
        "extra_args": ["--max-model-len", "4096"],
        "source": "vllm-manager",
    }
    merged = merge_runtime_payload(
        architecture_meta=arch_meta,
        runtime_payload=runtime_payload,
        timestamp="2024-01-01T00:00:00Z",
    )

    assert merged["runtime"]["runtime_context_tokens"] == 4096
    assert merged["runtime"]["extra_args_snapshot"] == ["--max-model-len", "4096"]
    assert merged["runtime"]["observed_at"] == "2024-01-01T00:00:00Z"
    assert "runtime-log" in merged["sources"]


def test_collect_architecture_from_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def show_model(self, name: str) -> dict:
            return {
                "details": {"context_length": 8192, "parameter_size": "8B"},
            }

        def close(self):
            pass

    monkeypatch.setattr(arch_module, "OllamaClient", lambda: DummyClient())
    result = collect_architecture_metadata(
        Path("/nonexistent/path"), raw_path="ollama://dummy/model"
    )
    assert result.fields["context_length"] == 8192
    assert result.fields["params_billion"] == 8.0
