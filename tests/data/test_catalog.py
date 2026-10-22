import pytest

from amber import config
from amber.data.catalog import domain_of, load_items, thumb_path


@pytest.mark.parametrize(
    "item_id",
    ["x:1", "c", "c:", "c:abc", "c:1.5", "c:-1", "c:../1", "a:1:2", "", "C:1"],
)
def test_domain_of_rejects_malformed_ids(item_id):
    with pytest.raises(ValueError):
        domain_of(item_id)
    with pytest.raises(ValueError):
        thumb_path(item_id)


def test_thumb_path_for_clothes(tmp_path):
    assert thumb_path("c:12", tmp_path) == tmp_path / "clothes" / "thumbs" / "12.jpg"


def test_default_data_dir_follows_config_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert thumb_path("c:3") == tmp_path / "clothes" / "thumbs" / "3.jpg"


def test_load_items_reads_each_domain(fixture_data):
    clothes = load_items("clothes", data_dir=fixture_data)
    art = load_items("art", data_dir=fixture_data)
    assert len(clothes) == 96 and clothes.id.iloc[0] == "c:1"
    assert len(art) == 24 and art.id.iloc[0] == "a:1"


def test_load_items_default_data_dir_follows_config(fixture_data, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", fixture_data)
    assert len(load_items("clothes")) == 96


def test_load_items_rejects_unknown_domain(fixture_data):
    with pytest.raises(ValueError):
        load_items("shoes", data_dir=fixture_data)  # type: ignore[arg-type]


def test_fixture_thumbnails_exist_for_every_item(fixture_data):
    for domain in ("clothes", "art"):
        for item_id in load_items(domain, data_dir=fixture_data).id:
            assert thumb_path(item_id, data_dir=fixture_data).exists()
