# Only import mono if cv2 is available (skip in TensorFlow container)
try:
    from .mono import check_monochrome as check_monochrome, MonoResult as MonoResult

    __all__ = ["check_monochrome", "MonoResult"]
except ImportError:
    # cv2 not available (e.g., TensorFlow container)
    __all__ = []
