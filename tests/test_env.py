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
    import faiss
    import open_clip
    import PIL
    import cv2

    # minimal “use” so Ruff doesn’t flag unused
    assert hasattr(faiss, "IndexFlatL2")
    assert callable(open_clip.create_model_and_transforms)
    assert hasattr(PIL, "Image")
    assert hasattr(cv2, "cvtColor")
