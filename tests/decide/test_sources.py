import json

import pandas as pd
import pytest

from amber.decide.events import EVENT_COLUMNS
from amber.decide.fake_sources import write_fake_sources
from amber.decide.sources import audit_lines, load_sources


@pytest.fixture(scope="module", params=["html", "json"])
def fake_root(request, tmp_path_factory):
    return write_fake_sources(tmp_path_factory.mktemp(request.param), fmt=request.param)


def test_load_sources_combines_everything(fake_root):
    df = load_sources(fake_root)
    assert list(df.columns) == EVENT_COLUMNS
    assert set(df["source"]) == {"google", "youtube", "instagram"}
    assert {"watch", "search", "visit", "discover", "like", "view_post", "ad_view"} <= set(df["kind"])
    assert df["ts"].is_monotonic_increasing
    assert df["from_ads"].any()
    assert not df.duplicated(["ts", "source", "kind", "item_id"]).any()
    for key in ("n_files", "n_skipped_files", "n_bad_rows", "n_skipped_rows"):
        assert key in df.attrs
    # Gemini folder and messages file were not read.
    assert df.attrs["n_skipped_files"] >= 2
    assert not df["text"].fillna("").str.contains("fake prompt|fake message").any()


def test_fake_watches_dedup_across_history_and_my_activity(fake_root):
    df = load_sources(fake_root)
    watches = df[(df["source"] == "youtube") & (df["kind"] == "watch") & df["item_id"].notna()]
    assert len(watches) == 400


def test_mojibake_fixed_in_fake_instagram(fake_root):
    df = load_sources(fake_root)
    creators = df.loc[df["source"] == "instagram", "creator_id"].dropna()
    assert creators.str.contains("fake_usér_").any()
    assert not creators.str.contains("Ã").any()


def test_fake_sources_deterministic_and_validated(tmp_path):
    a = write_fake_sources(tmp_path / "a", seed=3, scale=0.2)
    b = write_fake_sources(tmp_path / "b", seed=3, scale=0.2)
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert fa == fb
    assert all((a / p).read_bytes() == (b / p).read_bytes() for p in fa)
    with pytest.raises(ValueError):
        write_fake_sources(tmp_path / "c", fmt="xml")


def test_scans_meta_and_instagram_folders_and_dedups(tmp_path):
    rec = {"k": [{"title": "u", "string_list_data": [{"href": "h", "timestamp": 1_700_000_000}]}]}
    for folder in ("meta", "instagram"):
        f = tmp_path / folder / "liked_posts.json"
        f.parent.mkdir()
        f.write_text(json.dumps(rec))
    df = load_sources(tmp_path)
    assert len(df) == 1 and df.iloc[0]["kind"] == "like"


def test_empty_and_missing_folders_raise(tmp_path):
    with pytest.raises(ValueError, match="no events"):
        load_sources(tmp_path)
    (tmp_path / "google").mkdir()
    with pytest.raises(ValueError, match="no events"):
        load_sources(tmp_path)
    with pytest.raises(ValueError, match="not a folder"):
        load_sources(tmp_path / "nope")


def test_audit_lines_has_counts_and_no_private_text(fake_root):
    df = load_sources(fake_root)
    text = "\n".join(audit_lines(df))
    assert "youtube" in text and "watch" in text and "instagram" in text
    assert "active months" in text
    assert "skipped files" in text and "bad rows" in text
    for needle in ("Fake Video", "fake_user", "fake query", "example.com", "youtube.com", "fakevid"):
        assert needle not in text


def test_audit_active_months_and_dates():
    ts = pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-02"], utc=True)
    df = pd.DataFrame(
        {
            "ts": ts,
            "source": "google",
            "kind": "search",
            "item_id": ["a", "b", "c"],
            "creator_id": None,
            "text": None,
            "url": None,
            "from_ads": False,
        }
    )
    text = "\n".join(audit_lines(df))
    assert "2024-01-05" in text and "2024-03-02" in text
    row = next(line for line in text.splitlines() if line.startswith("google") and "2024-01-05" in line)
    assert row.split()[-1] == "2"  # January and March are active, February is not
