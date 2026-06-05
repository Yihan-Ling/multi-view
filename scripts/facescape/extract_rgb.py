"""Extract the real captured multi-view RGB for later use (deliverable #2).

Independent of the synthetic render: organizes the real photos of all *valid* views
(plus their params) for each selected subject into a clean layout:

    raw_rgb/<id>/<view>.jpg          # real captured photo (valid views only)
    raw_rgb/<id>/params.json         # copied camera parameters
    raw_rgb/<id>/manifest.json       # view list, dims, validity

With --undistort, also writes undistorted copies under raw_rgb/<id>/undistorted/
(the FaceScape models project onto the *undistorted* image; see doc_mview_model).

    .venv-data/bin/python scripts/data/extract_rgb.py [--undistort]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from multi_view.data import facescape as fs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(REPO / "data" / "facescape" / "raw"))
    ap.add_argument("--out-root", default=str(REPO / "data" / "facescape" / "raw_rgb"))
    ap.add_argument("--selection", default=str(REPO / "data" / "facescape" / "selection.json"))
    ap.add_argument("--exp", default="1_neutral")
    ap.add_argument("--undistort", action="store_true")
    args = ap.parse_args()

    ids = [s["id"] for s in json.load(open(args.selection))["subjects"]]
    mview = Path(args.raw_root) / "mview"

    for sid in ids:
        tuple_dir = mview / str(sid) / args.exp
        params_path = tuple_dir / "params.json"
        if not params_path.exists():
            print(f"id {sid}: no params.json at {tuple_dir} -- skip")
            continue
        params = fs.load_params(params_path)
        out = Path(args.out_root) / str(sid)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(params_path, out / "params.json")

        kept = []
        for vid in fs.view_ids(params, valid_only=True):
            jpg = tuple_dir / f"{vid}.jpg"
            if not jpg.exists():
                continue
            shutil.copy2(jpg, out / f"{vid}.jpg")
            kept.append(vid)
            if args.undistort:
                import cv2

                vp = fs.get_view(params, vid)
                img = cv2.imread(str(jpg))
                und = cv2.undistort(img, vp.K, vp.dist)
                (out / "undistorted").mkdir(exist_ok=True)
                cv2.imwrite(str(out / "undistorted" / f"{vid}.jpg"), und)

        (out / "manifest.json").write_text(
            json.dumps(
                {"subject": sid, "expression": args.exp, "valid_views": kept,
                 "n_views": len(kept), "undistorted": bool(args.undistort)},
                indent=2,
            )
        )
        print(f"id {sid}: extracted {len(kept)} valid RGB views -> {out}")


if __name__ == "__main__":
    main()
