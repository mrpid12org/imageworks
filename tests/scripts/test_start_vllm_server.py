from __future__ import annotations

from importlib import util
from pathlib import Path
import types


def load_module() -> types.ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "start_vllm_server.py"
    spec = util.spec_from_file_location("start_vllm_server", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_command_strips_remainder_sentinel(tmp_path):
    module = load_module()
    model_path = tmp_path / "model"
    model_path.mkdir()

    ns = types.SimpleNamespace(
        model=str(model_path),
        host="0.0.0.0",
        port=8000,
        served_model_name="demo",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        max_num_seqs=None,
        max_num_batched_tokens=None,
        kv_cache_dtype=None,
        swap_space=None,
        enforce_eager=False,
        trust_remote_code=False,
        api_keys=None,
        chat_template=None,
        extra=["--", "--enable-auto-tool-choice", "--tool-call-parser", "openai"],
        background=False,
        log_file=None,
    )

    command = module.build_command(ns)
    assert "--" not in command
    assert "--enable-auto-tool-choice" in command
    assert "--tool-call-parser" in command
    assert "openai" in command
    assert "--chat-template-content-format" in command
    assert command.index("--chat-template-content-format") < command.index(
        "--tool-call-parser"
    )
    assert "--return-tokens-as-token-ids" in command
    assert "--tokenizer-mode" in command
    assert command[command.index("--tokenizer-mode") + 1] == "slow"


def test_build_command_preserves_explicit_chat_format(tmp_path):
    module = load_module()
    model_path = tmp_path / "model"
    model_path.mkdir()

    ns = types.SimpleNamespace(
        model=str(model_path),
        host="0.0.0.0",
        port=8000,
        served_model_name="demo",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        max_num_seqs=None,
        max_num_batched_tokens=None,
        kv_cache_dtype=None,
        swap_space=None,
        enforce_eager=False,
        trust_remote_code=False,
        api_keys=None,
        chat_template=None,
        extra=[
            "--tool-call-parser",
            "openai",
            "--chat-template-content-format",
            "string",
        ],
        background=False,
        log_file=None,
    )

    command = module.build_command(ns)
    assert command.count("--chat-template-content-format") == 1
    assert command[command.index("--chat-template-content-format") + 1] == "string"


def test_build_command_handles_equals_style(tmp_path):
    module = load_module()
    model_path = tmp_path / "model"
    model_path.mkdir()

    ns = types.SimpleNamespace(
        model=str(model_path),
        host="0.0.0.0",
        port=8000,
        served_model_name="demo",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        max_num_seqs=None,
        max_num_batched_tokens=None,
        kv_cache_dtype=None,
        swap_space=None,
        enforce_eager=False,
        trust_remote_code=False,
        api_keys=None,
        chat_template=None,
        extra=["--tool-call-parser=openai"],
        background=False,
        log_file=None,
    )

    command = module.build_command(ns)
    assert "--chat-template-content-format" in command
    assert command[command.index("--chat-template-content-format") + 1] == "openai"
    assert "--return-tokens-as-token-ids" in command
    assert "--tokenizer-mode" in command


def test_build_command_respects_explicit_tokenizer_and_token_flags(tmp_path):
    module = load_module()
    model_path = tmp_path / "model"
    model_path.mkdir()

    ns = types.SimpleNamespace(
        model=str(model_path),
        host="0.0.0.0",
        port=8000,
        served_model_name="demo",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        max_num_seqs=None,
        max_num_batched_tokens=None,
        kv_cache_dtype=None,
        swap_space=None,
        enforce_eager=False,
        trust_remote_code=False,
        api_keys=None,
        chat_template=None,
        extra=[
            "--tool-call-parser",
            "openai",
            "--return-tokens-as-token-ids",
            "--tokenizer-mode",
            "fast",
        ],
        background=False,
        log_file=None,
    )

    command = module.build_command(ns)
    assert command.count("--return-tokens-as-token-ids") == 1
    tokenizer_idx = command.index("--tokenizer-mode")
    assert command[tokenizer_idx + 1] == "fast"
