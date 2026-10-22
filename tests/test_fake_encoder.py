import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import FakeEncoder


def _thumb(data_dir, domain, colour):
    items = pd.read_parquet(data_dir / domain / "items.parquet")
    n = items.loc[items.base_colour == colour, "id"].iloc[0].split(":")[1]
    return data_dir / domain / "thumbs" / f"{n}.jpg"


def test_fake_encoder_is_deterministic_and_unit(fake_encoder):
    a, b = fake_encoder.embed_texts(["hello", "hello"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_fake_encoder_protocol_attributes(fake_encoder):
    assert fake_encoder.dim == 16
    assert fake_encoder.model_name == "fake" and fake_encoder.pretrained == "fixture"
    assert FakeEncoder(dim=8).dim == 8


def test_fake_encoder_output_shapes_and_dtype(fake_encoder, fixture_data):
    paths = [
        _thumb(fixture_data, "clothes", "Red"),
        _thumb(fixture_data, "clothes", "Blue"),
    ]
    imgs = fake_encoder.embed_images(paths)
    txts = fake_encoder.embed_texts(["a", "b", "c"])
    assert imgs.shape == (2, 16) and imgs.dtype == np.float32
    assert txts.shape == (3, 16) and txts.dtype == np.float32
    assert np.allclose(np.linalg.norm(imgs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(txts, axis=1), 1.0, atol=1e-5)
    assert fake_encoder.embed_texts([]).shape == (0, 16)
    assert fake_encoder.embed_images([]).shape == (0, 16)


def test_fake_encoder_text_is_stable_across_instances_and_distinct_per_text(
    fake_encoder,
):
    text = "a photo of a piece of clothing"
    assert np.allclose(
        FakeEncoder(dim=16).embed_texts([text])[0], fake_encoder.embed_texts([text])[0]
    )
    assert not np.allclose(
        fake_encoder.embed_texts(["hello"])[0], fake_encoder.embed_texts(["goodbye"])[0]
    )


def test_fake_encoder_is_stable_across_processes(fixture_data):
    # builtin hash() is salted per process; the encoder must seed from hashlib instead
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from conftest import FakeEncoder\n"
        "e = FakeEncoder()\n"
        "print(e.embed_texts(['hello'])[0].tolist(), e.embed_images([sys.argv[2]])[0].tolist())\n"
    )
    thumb = _thumb(fixture_data, "clothes", "Red")
    outs = {
        subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).parent), str(thumb)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outs) == 1


def test_fake_encoder_colors(fake_encoder, fixture_data):
    # a red image is closer to "red" text than a blue image is
    red_img, blue_img = fake_encoder.embed_images(
        [
            _thumb(fixture_data, "clothes", "Red"),
            _thumb(fixture_data, "clothes", "Blue"),
        ]
    )
    t_red = fake_encoder.embed_texts(["red"])[0]
    assert np.dot(red_img, t_red) > np.dot(blue_img, t_red)
    t_blue = fake_encoder.embed_texts(["a blue shirt"])[0]
    assert np.dot(blue_img, t_blue) > np.dot(red_img, t_blue)


def test_fake_encoder_color_word_sets_dimension(fake_encoder):
    red, green, blue, none = fake_encoder.embed_texts(
        ["Red", "GREEN dress", "blue", "plain"]
    )
    assert red.argmax() == 0 and green.argmax() == 1 and blue.argmax() == 2
    assert red[0] > 0.9 and green[1] > 0.9 and blue[2] > 0.9
    assert none.max() < 0.9


def test_fake_encoder_image_dims_are_mean_rgb(fake_encoder, fixture_data):
    img = fake_encoder.embed_images([_thumb(fixture_data, "clothes", "Red")])[0]
    assert img[0] > img[1] and img[0] > img[2] and img[0] > 0.8
