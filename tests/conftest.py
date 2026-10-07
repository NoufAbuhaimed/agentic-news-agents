import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path):
    """Point the app at a temp folder so tests never touch the real data/ (settings is frozen)."""
    from digest import costs, nodes
    from digest.config import settings

    original = settings.data_dir
    object.__setattr__(settings, "data_dir", tmp_path)
    nodes.store.cache_clear()
    costs._current = None
    costs._search_down = None
    nodes.newsletter_leads = lambda since, **kw: []  # tests never call the real newsletter
    yield
    from digest import fetch
    nodes.newsletter_leads = fetch.newsletter_leads
    object.__setattr__(settings, "data_dir", original)
    nodes.store.cache_clear()
    costs._current = None
