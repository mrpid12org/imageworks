"""Tests for the SigLIP embedding service scaffold."""

from imageworks.libs.personal_tagger.embeddings import SiglipEmbeddingService


def test_encode_uses_custom_loader():
    calls = {}

    def loader(model_id: str):
        calls["model_id"] = model_id

        def encoder(texts):
            return [[float(len(item))] for item in texts]

        return encoder

    service = SiglipEmbeddingService(encoder_loader=loader)
    vectors = service.encode(["abc", "de"])

    assert calls["model_id"] == "google/siglip-large-patch16-384"
    assert vectors == [[3.0], [2.0]]


def test_encode_raises_when_downloader_missing(monkeypatch):
    def loader(model_id: str):
        raise RuntimeError("dependencies missing")

    service = SiglipEmbeddingService(encoder_loader=loader)

    try:
        service.encode(["hello"])
    except RuntimeError as exc:
        assert "dependencies" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected runtime error when loader fails")
