"""Shared filesystem locations.

Every script and notebook resolves data through here, so the repo runs from any
checkout. Set AI_PRODUCTIVITY_DATA to point at a data directory elsewhere
(for example the full private caches), otherwise the repo's own data/ is used.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AI_PRODUCTIVITY_DATA", PROJECT_ROOT / "data")).resolve()

# Private material that must never be committed: raw caches with commit
# messages, the login list, and the pseudonymisation salt.
PRIVATE_DIR = Path(os.environ.get("AI_PRODUCTIVITY_PRIVATE", PROJECT_ROOT / "private")).resolve()
