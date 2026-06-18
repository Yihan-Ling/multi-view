"""Patch HRNet's lib/utils/transforms.py for modern scipy/numpy.

The upstream crop() uses scipy.misc.imresize / imrotate (removed in scipy>=1.12)
and np.math.floor (removed in numpy>=2.0), so training crashes on the first batch.
This rewrites crop() to warp with cv2 using the SAME get_transform matrix that
transform_pixel uses -- so the image and the 68 landmarks stay aligned (mixing in
the repo's cv2 crop_v2, which uses a different transform, would misalign them).

Idempotent: re-running is a no-op (guarded by the FACESCAPE_CV2_CROP marker).
Run from the main repo root:  python scripts/facescape/hrnet/fix_transforms.py
"""
import re
from pathlib import Path

TRANSFORMS = Path("third_party/HRNet-Facial-Landmark-Detection/lib/utils/transforms.py")

NEW_CROP = '''def crop(img, center, scale, output_size, rot=0):  # FACESCAPE_CV2_CROP
    """cv2 replacement for the original scipy.misc crop. Uses the SAME
    get_transform matrix as transform_pixel, so the image and the landmarks
    stay aligned. Handles rotation via the matrix directly."""
    import numpy as np
    import cv2
    if hasattr(center, "numpy"):
        center = center.numpy()
    t = get_transform(center, scale, output_size, rot=rot)[:2].astype(np.float32)
    return cv2.warpAffine(
        np.asarray(img), t,
        (int(output_size[1]), int(output_size[0])),
        flags=cv2.INTER_LINEAR,
    )
'''


def main():
    if not TRANSFORMS.exists():
        raise SystemExit(f"not found: {TRANSFORMS} (clone HRNet first -- see README)")
    src = TRANSFORMS.read_text()
    if "FACESCAPE_CV2_CROP" in src:
        print("already patched -- nothing to do")
        return
    # drop the scipy imports (scipy.misc no longer exists)
    src = src.replace("import scipy\nimport scipy.misc\n", "")
    # replace the whole crop() body (crop_v2 is left untouched: 'def crop\\(' won't match it)
    src, n = re.subn(
        r"def crop\(img, center, scale, output_size, rot=0\):.*?(?=\ndef |\Z)",
        NEW_CROP, src, count=1, flags=re.S,
    )
    if n != 1:
        raise SystemExit("could not locate crop() to replace -- upstream may have changed")
    TRANSFORMS.write_text(src)
    print("patched crop() + removed scipy imports in transforms.py")


if __name__ == "__main__":
    main()
