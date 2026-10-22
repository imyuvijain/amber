"""CLI commands that work on the embedding space: embed, eval-space, search-text,
shrink-thumbs. The real CLIP model is swapped for the model-free FakeEncoder."""

import pytest
from conftest import FakeEncoder
from PIL import Image
from typer.testing import CliRunner

from amber import config
from amber.cli import app
from amber.space.index import VectorIndex, embed_dir


@pytest.fixture
def fake_clip(monkeypatch, fixture_data):
    """Point DATA_DIR at the fixture data and make `ClipEncoder(...)` build a FakeEncoder."""
    made = []

    def factory(model_name, pretrained, **kwargs):
        made.append((model_name, pretrained))
        return FakeEncoder(dim=16)

    monkeypatch.setattr(config, "DATA_DIR", fixture_data)
    monkeypatch.setattr("amber.space.encoder.ClipEncoder", factory)
    return made


def _emb(fixture_data, domain):
    return fixture_data / domain / "emb" / "fake__fixture"


def test_embed_defaults_to_the_configured_model_and_all_domains(
    fake_clip, fixture_data
):
    result = CliRunner().invoke(app, ["embed"])
    assert result.exit_code == 0, result.output
    assert fake_clip == [(config.MODEL_NAME, config.MODEL_PRETRAINED)]  # loaded once
    assert len(VectorIndex.load(_emb(fixture_data, "clothes")).ids) == 96
    assert len(VectorIndex.load(_emb(fixture_data, "art")).ids) == 24
    assert "96" in result.output and "24" in result.output


def test_embed_takes_model_pretrained_and_one_domain(fake_clip, fixture_data):
    args = ["embed", "--model", "M", "--pretrained", "P", "--domain", "art"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert fake_clip == [("M", "P")]
    assert _emb(fixture_data, "art").exists()
    assert not (fixture_data / "clothes" / "emb").exists()


def test_embed_all_skips_a_domain_that_was_not_ingested(fake_clip, fixture_data):
    (fixture_data / "art" / "items.parquet").unlink()
    result = CliRunner().invoke(app, ["embed", "--domain", "all"])
    assert result.exit_code == 0, result.output
    assert _emb(fixture_data, "clothes").exists()
    assert "art" in result.output and "skip" in result.output.lower()


def test_embed_rejects_an_unknown_domain(fake_clip):
    assert CliRunner().invoke(app, ["embed", "--domain", "shoes"]).exit_code != 0


def test_eval_space_prints_precision(fake_clip, fixture_data):
    CliRunner().invoke(app, ["embed", "--domain", "clothes"])
    args = ["eval-space", "--model", "M", "--pretrained", "P", "--min-group", "2"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "P@20" in result.output and "dim 16" in result.output


def test_eval_space_with_no_group_big_enough_explains(fake_clip):
    CliRunner().invoke(app, ["embed", "--domain", "clothes"])
    result = CliRunner().invoke(app, ["eval-space"])  # fixture groups are tiny
    assert result.exit_code != 0 and "min-group" in result.output


def test_eval_space_without_embeddings_says_to_run_embed(fake_clip):
    result = CliRunner().invoke(app, ["eval-space"])
    assert result.exit_code != 0 and "amber embed" in result.output


def test_search_text_writes_a_contact_sheet_of_the_best_matches(
    fake_clip, fixture_data
):
    CliRunner().invoke(app, ["embed"])
    result = CliRunner().invoke(app, ["search-text", "a red dress"])
    assert result.exit_code == 0, result.output
    sheet = fixture_data / "contact_sheet.jpg"
    with Image.open(sheet) as im:
        assert im.format == "JPEG" and im.size == (4 * 216, 3 * 288)
    # FakeEncoder points "red" along dim 0: the top hit is one of the clear reds
    first = next(line for line in result.output.splitlines() if "c:" in line)
    assert "Red" in first


def test_search_text_can_search_art(fake_clip, fixture_data):
    CliRunner().invoke(app, ["embed"])
    result = CliRunner().invoke(app, ["search-text", "red", "--domain", "art"])
    assert result.exit_code == 0, result.output
    assert "a:1" in result.output  # a:1 is the red artwork
    assert (fixture_data / "contact_sheet.jpg").exists()


def test_search_text_without_embeddings_says_to_run_embed(fake_clip):
    result = CliRunner().invoke(app, ["search-text", "red"])
    assert result.exit_code != 0 and "amber embed" in result.output


def test_embed_dir_matches_what_the_cli_writes(fake_clip, fixture_data):
    CliRunner().invoke(app, ["embed", "--domain", "art"])
    assert embed_dir("art", FakeEncoder(), fixture_data) == _emb(fixture_data, "art")


def test_shrink_thumbs_shrinks_clothes_in_place(monkeypatch, fixture_data):
    monkeypatch.setattr(config, "DATA_DIR", fixture_data)
    result = CliRunner().invoke(app, ["shrink-thumbs"])
    assert result.exit_code == 0, result.output
    with Image.open(fixture_data / "clothes" / "thumbs" / "1.jpg") as im:
        assert im.size == config.THUMB_SIZE
    assert "96" in result.output
    again = CliRunner().invoke(app, ["shrink-thumbs", "--domain", "clothes"])
    assert again.exit_code == 0 and "0" in again.output
