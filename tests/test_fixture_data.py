import numpy as np
import pandas as pd
from conftest import build_fixture_data
from PIL import Image

CLOTHES_COLUMNS = [
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
ART_COLUMNS = ["id", "title", "artist", "date", "artwork_type"]


def test_clothes_fixture_layout(fixture_data):
    df = pd.read_parquet(fixture_data / "clothes" / "items.parquet")
    assert list(df.columns) == CLOTHES_COLUMNS
    assert df.id.tolist() == [f"c:{n}" for n in range(1, 97)]
    counts = df.groupby(["gender", "master_category"]).size()
    assert counts.to_dict() == {
        (g, s): 24 for g in ("Men", "Women") for s in ("Apparel", "Footwear")
    }
    assert (df[CLOTHES_COLUMNS[1:9]] != "").all().all()
    for item_id in df.id:
        n = item_id.split(":")[1]
        assert (fixture_data / "clothes" / "thumbs" / f"{n}.jpg").is_file()


def test_every_clothes_group_has_clear_red_and_blue(fixture_data):
    df = pd.read_parquet(fixture_data / "clothes" / "items.parquet")
    for _, group in df.groupby(["gender", "master_category"]):
        assert {"Red", "Blue"} <= set(group.base_colour)
        for colour, channel in (("Red", 0), ("Blue", 2)):
            n = group.loc[group.base_colour == colour, "id"].iloc[0].split(":")[1]
            with Image.open(fixture_data / "clothes" / "thumbs" / f"{n}.jpg") as im:
                px = im.convert("RGB").getpixel((0, 0))
            others = [c for i, c in enumerate(px) if i != channel]
            assert px[channel] > 180 and max(others) < 80


def test_clothes_thumbs_are_small_and_colors_distinct(fixture_data):
    df = pd.read_parquet(fixture_data / "clothes" / "items.parquet")
    colours = set()
    for item_id in df.id:
        n = item_id.split(":")[1]
        with Image.open(fixture_data / "clothes" / "thumbs" / f"{n}.jpg") as im:
            assert im.size == (48, 64)
            colours.add(im.convert("RGB").getpixel((0, 0)))
    assert len(colours) > 80  # distinct up to JPEG rounding


def test_art_fixture_layout(fixture_data):
    df = pd.read_parquet(fixture_data / "art" / "items.parquet")
    assert list(df.columns) == ART_COLUMNS
    assert df.id.tolist() == [f"a:{n}" for n in range(1, 25)]
    assert set(df.artwork_type) <= {
        "Painting",
        "Photograph",
        "Print",
        "Drawing and Watercolor",
    }
    assert (df != "").all().all()
    for item_id in df.id:
        n = item_id.split(":")[1]
        assert (fixture_data / "art" / "thumbs" / f"{n}.jpg").is_file()


def test_art_fixture_has_red_and_blue(fixture_data, fake_encoder):
    paths = [fixture_data / "art" / "thumbs" / f"{n}.jpg" for n in range(1, 25)]
    imgs = fake_encoder.embed_images(paths)
    assert (imgs[:, 0] > 0.8).any() and (imgs[:, 2] > 0.8).any()
    assert np.isfinite(imgs).all()


def test_fixture_data_is_deterministic(tmp_path):
    a = build_fixture_data(tmp_path / "a")
    b = build_fixture_data(tmp_path / "b")
    for domain in ("clothes", "art"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(a / domain / "items.parquet"),
            pd.read_parquet(b / domain / "items.parquet"),
        )
        for thumb in (a / domain / "thumbs").iterdir():
            assert (
                thumb.read_bytes() == (b / domain / "thumbs" / thumb.name).read_bytes()
            )


def test_fixture_data_lives_under_tmp_path(fixture_data, tmp_path):
    assert fixture_data.is_dir() and tmp_path in fixture_data.parents
