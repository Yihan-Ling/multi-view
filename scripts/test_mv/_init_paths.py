import sys
from pathlib import Path

# scripts/test_mv/_init_paths.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
