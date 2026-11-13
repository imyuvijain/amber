import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def realish():
    """Copy a fake export with every track uri and name rewritten to look real.

    Used to exercise the paths that need non-synthetic data (e.g. publishing).
    """

    def convert(src: Path, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.json"):
            text = f.read_text()
            text = re.sub(
                r"spotify:track:fake(\d+)x(\d+)",
                lambda m: f"spotify:track:{int(m[1]):011d}{int(m[2]):011d}",
                text,
            )
            text = text.replace("Fake ", "Some ")
            json.loads(text)
            (dest / f.name).write_text(text)
        return dest

    return convert
