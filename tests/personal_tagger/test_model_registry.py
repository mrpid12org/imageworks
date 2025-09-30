"""Tests for the personal tagger model registry."""

from imageworks.apps.personal_tagger.core import model_registry


def test_list_models_default_returns_all():
    items = list(model_registry.list_models())
    assert items  # ensure registry not empty
    keys = {variant.key for variant in items}
    assert "caption-qwen2.5-vl-7b-awq" in keys


def test_list_models_filtered_by_stage():
    caption_variants = list(model_registry.list_models("caption"))
    assert caption_variants
    for variant in caption_variants:
        assert variant.key.startswith("caption")


def test_get_model_known_key():
    variant = model_registry.get_model("description-idefics2-8b")
    assert variant.display_name.startswith("Idefics2")


def test_get_model_unknown_key_raises():
    try:
        model_registry.get_model("missing")
    except ValueError as exc:
        assert "Unknown model key" in str(exc)
    else:  # pragma: no cover - fail fast
        raise AssertionError("Expected ValueError for missing key")
