from imageworks.libs.personal_tagger.embeddings import SiglipEmbeddingService


def test_siglip_embedding_service_lazy_init_dummy_loader():
    # Provide a dummy encoder loader so we don't require transformers/torch for the test
    calls = {}

    def dummy_loader(_model_id: str):  # returns encoder fn
        def _encode(texts):
            calls["invoked"] = True
            # Return fixed-length zero vectors for determinism
            return [[0.0, 0.0, 0.0] for _ in texts]

        return _encode

    svc = SiglipEmbeddingService(encoder_loader=dummy_loader)
    result = svc.encode(["a", "b"])
    assert calls.get("invoked") is True
    assert len(result) == 2
    assert all(len(vec) == 3 for vec in result)
