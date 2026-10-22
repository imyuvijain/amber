import io
import logging
import os
import subprocess
import sys

import pandas as pd
import pytest
from PIL import Image

from amber.data.catalog import domain_of, load_items, thumb_path
from amber.data.clothes import CLOTHES_DATASET, ingest_clothes

COLUMNS = [
    "id",
    "title",
    "gender",
    "master_category",
    "sub_category",
    "article_type",
    "base_colour",
    "season",
    "usage",
    "year",
]


class _BrokenImage:
    """Stands in for an image whose pixels cannot be decoded."""

    def convert(self, mode):
        raise OSError("image file is truncated")


def _solid(rgb, size=(96, 128)):
    return Image.new("RGB", size, rgb)


def _jpeg_bytes(rgb, size=(96, 128)):
    buf = io.BytesIO()
    _solid(rgb, size).save(buf, "JPEG")
    return buf.getvalue()


def _row(item_id, master, image, **overrides):
    row = {
        "id": item_id,
        "gender": "Men",
        "masterCategory": master,
        "subCategory": "Topwear",
        "articleType": "Tshirts",
        "baseColour": "Red",
        "season": "Summer",
        "year": 2012.0,
        "usage": "Casual",
        "productDisplayName": f"Product {item_id}",
        "image": image,
    }
    row.update(overrides)
    return row


@pytest.fixture
def fake_rows():
    """Five rows shaped like the Hugging Face ones (ids 1-5).

    1 Apparel (PIL image), 2 Footwear (`datasets.Image(decode=False)` dict, landscape),
    3 Accessories (missing strings and a NaN year), 4 Personal Care (filtered out),
    5 Apparel whose image cannot be decoded (skipped).
    """
    return [
        _row(1, "Apparel", _solid((220, 20, 20))),
        _row(
            2,
            "Footwear",
            {"bytes": _jpeg_bytes((20, 20, 220), (200, 100)), "path": None},
            gender="Women",
            subCategory="Shoes",
            articleType="Heels",
            baseColour="Blue",
        ),
        _row(
            3,
            "Accessories",
            _solid((20, 200, 20)),
            gender="Unisex",
            subCategory="Bags",
            articleType="Backpacks",
            baseColour=None,
            season=None,
            usage=None,
            year=float("nan"),
            productDisplayName=None,
        ),
        _row(4, "Personal Care", _solid((200, 200, 200)), subCategory="Fragrance"),
        _row(5, "Apparel", _BrokenImage()),
    ]


def test_ingest_filters_and_writes(tmp_path, fake_rows):
    df = ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert len(df) == 3 and df.id.str.startswith("c:").all()
    assert set(df.master_category) <= {"Apparel", "Footwear", "Accessories"}
    for item_id in df.id:
        with Image.open(thumb_path(item_id, data_dir=tmp_path)) as im:
            assert im.size == (288, 384)


def test_ingest_skips_corrupt_image(tmp_path, fake_rows):
    df = ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert "c:5" not in set(df.id)  # the corrupt row's id
    assert not (tmp_path / "clothes" / "thumbs" / "5.jpg").exists()


def test_ingest_logs_corrupt_image(tmp_path, fake_rows, caplog):
    with caplog.at_level(logging.WARNING, logger="amber.data.clothes"):
        ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert any("c:5" in r.getMessage() for r in caplog.records)


def test_ingest_skips_undecodable_bytes(tmp_path, fake_rows):
    fake_rows[0]["image"] = {"bytes": b"definitely not an image", "path": None}
    fake_rows[1]["image"] = {"bytes": None, "path": None}
    df = ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert list(df.id) == ["c:3"]
    assert not (tmp_path / "clothes" / "thumbs" / "1.jpg").exists()
    assert not (tmp_path / "clothes" / "thumbs" / "2.jpg").exists()


def test_ingest_resumes(tmp_path, fake_rows):
    out = tmp_path / "clothes"
    ingest_clothes(out, rows=fake_rows)
    mtimes = {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()}
    df2 = ingest_clothes(out, rows=fake_rows)
    assert {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()} == mtimes
    assert len(df2) == 3


def test_ingest_resume_recovers_rows_when_parquet_is_missing(tmp_path, fake_rows):
    # an interrupted run can leave thumbnails but no parquet: those rows must not be lost
    out = tmp_path / "clothes"
    ingest_clothes(out, rows=fake_rows)
    (out / "items.parquet").unlink()
    mtimes = {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()}
    df = ingest_clothes(out, rows=fake_rows)
    assert list(df.id) == ["c:1", "c:2", "c:3"]
    assert {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()} == mtimes
    assert list(pd.read_parquet(out / "items.parquet").id) == ["c:1", "c:2", "c:3"]


def test_ingest_merges_with_existing_parquet_without_duplicates(tmp_path, fake_rows):
    out = tmp_path / "clothes"
    first = ingest_clothes(out, rows=fake_rows[:2])
    assert list(first.id) == ["c:1", "c:2"]
    df = ingest_clothes(out, rows=fake_rows[1:])  # c:2 is seen again, c:3 is new
    assert list(df.id) == ["c:1", "c:2", "c:3"]
    assert df.id.is_unique
    assert list(pd.read_parquet(out / "items.parquet").id) == ["c:1", "c:2", "c:3"]


def test_ingest_parquet_has_expected_columns_and_values(tmp_path, fake_rows):
    out = tmp_path / "clothes"
    ingest_clothes(out, rows=fake_rows)
    df = pd.read_parquet(out / "items.parquet")
    assert list(df.columns) == COLUMNS
    first = df.set_index("id").loc["c:1"]
    assert first.title == "Product 1" and first.gender == "Men"
    assert first.master_category == "Apparel" and first.sub_category == "Topwear"
    assert first.article_type == "Tshirts" and first.base_colour == "Red"
    assert first.season == "Summer" and first.usage == "Casual" and first.year == 2012
    third = df.set_index("id").loc[
        "c:3"
    ]  # missing strings become "", a NaN year stays null
    assert (third.title, third.base_colour, third.season, third.usage) == ("",) * 4
    assert pd.isna(third.year)
    assert load_items("clothes", data_dir=tmp_path).equals(df)


def test_ingest_thumbnails_are_jpeg_at_thumb_size_for_any_aspect_ratio(
    tmp_path, fake_rows
):
    ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    for n in (1, 2, 3):  # row 2 is a 200x100 landscape source
        with Image.open(tmp_path / "clothes" / "thumbs" / f"{n}.jpg") as im:
            assert im.format == "JPEG" and im.size == (288, 384)
    leftovers = [p.name for p in (tmp_path / "clothes" / "thumbs").iterdir()]
    assert sorted(leftovers) == ["1.jpg", "2.jpg", "3.jpg"]  # no temp files


def test_ingest_limit_counts_kept_items_and_stops_reading(tmp_path, fake_rows):
    pulled = []

    def stream():
        for row in fake_rows:
            pulled.append(row["id"])
            yield row

    df = ingest_clothes(tmp_path / "clothes", rows=stream(), limit=2)
    assert list(df.id) == ["c:1", "c:2"]
    assert pulled == [1, 2]
    with pytest.raises(ValueError):
        ingest_clothes(tmp_path / "clothes", rows=fake_rows, limit=0)


def test_ingest_empty_stream_writes_an_empty_catalog(tmp_path):
    df = ingest_clothes(tmp_path / "clothes", rows=[])
    assert len(df) == 0 and list(df.columns) == COLUMNS
    assert len(load_items("clothes", data_dir=tmp_path)) == 0


def test_ingest_streams_the_hub_dataset_when_no_rows_given(
    tmp_path, fake_rows, monkeypatch
):
    import datasets

    calls = {}

    class FakeStream:
        def cast_column(self, name, feature):
            calls["cast"] = (name, feature)
            return fake_rows

    def fake_load_dataset(name, **kwargs):
        calls["load"] = (name, kwargs)
        return FakeStream()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    df = ingest_clothes(tmp_path / "clothes")
    assert calls["load"] == (CLOTHES_DATASET, {"split": "train", "streaming": True})
    name, feature = calls["cast"]  # undecoded, so corrupt images fail per row
    assert name == "image" and isinstance(feature, datasets.Image)
    assert feature.decode is False
    assert len(df) == 3


def test_domain_of_and_thumb_path(tmp_path):
    assert domain_of("c:12") == "clothes" and domain_of("a:9") == "art"
    assert thumb_path("a:9", data_dir=tmp_path) == tmp_path / "art" / "thumbs" / "9.jpg"
    with pytest.raises(ValueError):
        domain_of("x:1")


@pytest.mark.slow
def test_real_stream_first_rows(tmp_path):
    # Real Hugging Face rows through the real CLI. A subprocess, because stopping a
    # pyarrow stream half-way is only clean when the CLI exits without interpreter teardown.
    proc = subprocess.run(
        [sys.executable, "-c", "from amber.cli import app; app()"]
        + ["ingest", "clothes", "--limit", "5"],
        env={**os.environ, "AMBER_DATA_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0] == "5"
    df = load_items("clothes", data_dir=tmp_path)
    assert len(df) == 5 and list(df.columns) == COLUMNS
    assert df.master_category.isin(["Apparel", "Footwear", "Accessories"]).all()
    for item_id in df.id:
        with Image.open(thumb_path(item_id, data_dir=tmp_path)) as im:
            assert im.size == (288, 384)
