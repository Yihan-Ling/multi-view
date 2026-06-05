"""Prune a downloaded FaceScape tree to just the selected subjects (neutral).

Copies (or symlinks) the files the render pipeline needs into data/facescape/raw/:

    raw/mview/<id>/<exp>/   <- params.json + all view .jpg  (from multi-view IMGS_ROOT)
    raw/tu/<id>/<exp>.obj   <- TU model .obj + .mtl + .jpg  (from TU models_reg)

This is the post-download step referenced in the plan: if the multi-view share is
only available as ID-range archives, download a range, run this to keep the selected
subjects, then delete the range to reclaim space.

Layout of the FaceScape download varies; pass the actual roots and verify the first
subject copied correctly:

    .venv-data/bin/python scripts/data/extract_selection.py \
        --mview-imgs /downloads/fsmview/images \
        --tu-models  /downloads/facescape_tu/models_reg \
        --selection  data/facescape/selection.json [--link]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def find_first(candidates: list[Path]) -> Path | None:
    return next((c for c in candidates if c.exists()), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mview-imgs", required=True, help="FaceScape multi-view IMGS_ROOT")
    ap.add_argument("--tu-models", required=True, help="FaceScape TU models_reg root")
    ap.add_argument("--selection", default=str(REPO / "data" / "facescape" / "selection.json"))
    ap.add_argument("--exp", default="1_neutral")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    ap.add_argument("--raw-root", default=str(REPO / "data" / "facescape" / "raw"))
    args = ap.parse_args()

    sel = json.load(open(args.selection))
    ids = [s["id"] for s in sel["subjects"]]
    mview_root, tu_root, raw = Path(args.mview_imgs), Path(args.tu_models), Path(args.raw_root)

    for sid in ids:
        # multi-view tuple (params.json + view jpgs)
        src_tuple = mview_root / str(sid) / args.exp
        if src_tuple.is_dir():
            for f in sorted(src_tuple.iterdir()):
                if f.suffix in (".jpg", ".json"):
                    place(f, raw / "mview" / str(sid) / args.exp / f.name, args.link)
            print(f"id {sid}: mview tuple copied ({len(list(src_tuple.glob('*.jpg')))} views)")
        else:
            print(f"id {sid}: WARNING multi-view tuple not found at {src_tuple}")

        # TU model (.obj + .mtl + .jpg) -- try a few common layouts
        obj = find_first([
            tu_root / str(sid) / f"{args.exp}.obj",
            tu_root / f"{sid}_{args.exp}.obj",
            tu_root / str(sid) / "models_reg" / f"{args.exp}.obj",
        ])
        if obj is None:
            print(f"id {sid}: WARNING TU .obj not found under {tu_root} -- check layout")
            continue
        for ext in (".obj", ".obj.mtl", ".jpg"):
            f = obj.with_suffix("").with_suffix(ext) if ext == ".obj.mtl" else obj.with_suffix(ext)
            f = obj.parent / (obj.stem + ext) if ext == ".obj.mtl" else f
            if f.exists():
                place(f, raw / "tu" / str(sid) / (obj.stem + ext), args.link)
        print(f"id {sid}: TU model copied from {obj.parent}")

    print(f"\ndone -> {raw}")


if __name__ == "__main__":
    main()
