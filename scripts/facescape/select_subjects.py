"""Select a stratified subset of FaceScape subjects to download/render.

Selection policy (see docs/data/facescape_rgbd.md):
  1. Publishable first -- only subjects on the login-gated ``publishable_list`` may
     appear in paper figures, so the candidate pool is restricted to it.
  2. Demographic stratification -- balance gender and spread age bands using
     ``info_list_v2.txt`` (index, gender, age, valid-label), to avoid skew within
     FaceScape's Asian-only population.

Both files come from the FaceScape download page after login (they ship with the
TU-model package). Point this script at them:

    .venv-data/bin/python scripts/data/select_subjects.py \
        --info-list  /path/to/info_list_v2.txt \
        --publishable-list /path/to/publishable_list.txt \
        --n 10 --out data/facescape/selection.json
"""

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGE_BANDS = [(16, 25), (26, 35), (36, 50), (51, 70)]


def parse_info_list(path: Path) -> dict[int, dict]:
    """Parse 'index gender age valid' rows ('-' means missing)."""
    info: dict[int, dict] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        idx = int(parts[0])
        gender = parts[1] if parts[1] in ("m", "f") else None
        age = int(parts[2]) if parts[2].isdigit() else None
        valid = parts[3] if len(parts) > 3 else "1"
        # first valid digit == model present/usable
        ok = valid[0] == "1" if valid and valid[0] in "01" else True
        info[idx] = {"gender": gender, "age": age, "valid": ok}
    return info


def parse_id_list(path: Path) -> set[int]:
    ids: set[int] = set()
    for tok in Path(path).read_text().replace(",", " ").split():
        if tok.isdigit():
            ids.add(int(tok))
    return ids


def age_band(age: int | None) -> int:
    if age is None:
        return len(AGE_BANDS)  # unknown bucket last
    for i, (lo, hi) in enumerate(AGE_BANDS):
        if lo <= age <= hi:
            return i
    return len(AGE_BANDS)


def stratified_pick(pool: list[int], info: dict, n: int, seed: int) -> list[int]:
    """Round-robin across (gender, age-band) buckets for a diverse subset."""
    rng = random.Random(seed)
    buckets: dict[tuple, list[int]] = {}
    for sid in pool:
        key = (info[sid]["gender"], age_band(info[sid]["age"]))
        buckets.setdefault(key, []).append(sid)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets, key=lambda k: (str(k[0]), k[1]))
    rng.shuffle(keys)
    picked: list[int] = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop())
                if len(picked) == n:
                    break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--info-list", required=True)
    ap.add_argument("--publishable-list", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "data" / "facescape" / "selection.json"))
    args = ap.parse_args()

    info = parse_info_list(Path(args.info_list))
    publishable = parse_id_list(Path(args.publishable_list))
    pool = sorted(sid for sid in publishable if info.get(sid, {}).get("valid", False))
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} valid publishable subjects, need {args.n}")

    picked = stratified_pick(pool, info, args.n, args.seed)
    subjects = [
        {"id": sid, "gender": info[sid]["gender"], "age": info[sid]["age"]}
        for sid in sorted(picked)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "criteria": {
                    "publishable_only": True,
                    "stratified_by": ["gender", "age_band"],
                    "age_bands": AGE_BANDS,
                    "candidate_pool_size": len(pool),
                    "seed": args.seed,
                },
                "subjects": subjects,
            },
            indent=2,
        )
    )
    print(f"selected {len(subjects)} subjects -> {out}")
    for s in subjects:
        print(f"  id {s['id']:>3}  {s['gender']}  age {s['age']}")


if __name__ == "__main__":
    main()
