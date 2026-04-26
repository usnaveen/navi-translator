"""Root conftest — adds project root to sys.path so `pytest` can import `src.*`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
