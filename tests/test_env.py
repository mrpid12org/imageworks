import os
import pytest
import torch


@pytest.mark.skipif(
    not (os.getenv("REQUIRE_CUDA") == "1"),
    reason="CUDA not required by default; set REQUIRE_CUDA=1 to enforce",
)
def test_torch_cuda():
    assert torch.cuda.is_available(), "CUDA not available to PyTorch"
    x = torch.rand(2, 3, device="cuda")
    assert x.is_cuda


def test_imports():
    try:
        import faiss  # type: ignore
    except Exception:  # noqa: BLE001
        faiss = None  # type: ignore
    try:
        import open_clip  # type: ignore
    except Exception:  # noqa: BLE001
        open_clip = None  # type: ignore
    import PIL
    import cv2

    if faiss is not None:  # only assert if available in environment
        assert hasattr(faiss, "IndexFlatL2")
    if open_clip is not None:
        assert callable(open_clip.create_model_and_transforms)
    assert hasattr(PIL, "Image")
    assert hasattr(cv2, "cvtColor")
