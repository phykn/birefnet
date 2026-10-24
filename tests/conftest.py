import os
import sys
from pathlib import Path

# Scripts draw with matplotlib; tests render off-screen instead of opening windows.
os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
