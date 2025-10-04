from imageworks.model_loader.naming import build_identity


def test_identity_slug_and_display_awq():
    identity = build_identity(
        family="qwen2.5-vl-7b-instruct",
        backend="vllm",
        format_type="awq",
        quantization="Q4_K_M",
    )
    assert identity.slug == "qwen2.5-vl-7b-instruct-vllm-awq-q4_k_m"
    assert identity.display_name == "Qwen2.5 VL 7B Instruct (AWQ Q4 K M, vLLM)"


def test_identity_backend_switch():
    base = build_identity(
        family="pixtral-12b-captioner-relaxed",
        backend="ollama",
        format_type="gguf",
        quantization="q4_0",
    )
    assert base.slug == "pixtral-12b-captioner-relaxed-ollama-gguf-q4_0"
    alt = base.with_backend("vllm")
    assert alt.slug == "pixtral-12b-captioner-relaxed-vllm-gguf-q4_0"
    assert base.display_name.startswith("Pixtral 12B Captioner Relaxed")


def test_identity_without_quantization():
    identity = build_identity(
        family="granite-3.3-8b-instruct",
        backend="vllm",
        format_type="safetensors",
        quantization=None,
    )
    assert identity.slug == "granite-3.3-8b-instruct-vllm-safetensors"
    assert identity.display_name == "Granite 3.3 8B Instruct (Safetensors, vLLM)"
