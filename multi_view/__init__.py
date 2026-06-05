"""multi_view package.

The model classes (``RGBDPoseResNet50``, ``MLPLandmarkHead``,
``SingleViewLandmarkModel``) are imported lazily so that lightweight subpackages
such as ``multi_view.data`` can be used in an environment without torch (e.g. the
FaceScape render venv).
"""

_LAZY = {
    "RGBDPoseResNet50": "multi_view.backbone",
    "MLPLandmarkHead": "multi_view.head",
    "SingleViewLandmarkModel": "multi_view.model",
}

__all__ = list(_LAZY)


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
