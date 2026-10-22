from PIL import Image

from amber.data.files import shrink_thumbnails


def _write(path, size, quality=85):
    Image.new("RGB", size, (200, 30, 30)).save(path, "JPEG", quality=quality)


def test_shrink_resizes_every_thumbnail_and_reports_how_many(tmp_path):
    for n in (1, 2, 3):
        _write(tmp_path / f"{n}.jpg", (288, 384))
    assert shrink_thumbnails(tmp_path, (216, 288), 80) == 3
    for n in (1, 2, 3):
        with Image.open(tmp_path / f"{n}.jpg") as im:
            assert im.format == "JPEG" and im.size == (216, 288)


def test_shrink_is_idempotent_and_leaves_matching_files_alone(tmp_path):
    _write(tmp_path / "1.jpg", (288, 384))
    _write(tmp_path / "2.jpg", (216, 288), quality=95)
    before = (tmp_path / "2.jpg").read_bytes()
    assert shrink_thumbnails(tmp_path, (216, 288), 80) == 1
    assert (tmp_path / "2.jpg").read_bytes() == before  # already at size: untouched
    assert shrink_thumbnails(tmp_path, (216, 288), 80) == 0  # second run does nothing


def test_shrink_uses_the_given_jpeg_quality_and_gets_smaller(tmp_path):
    noisy = Image.effect_noise((288, 384), 60).convert("RGB")
    noisy.save(tmp_path / "1.jpg", "JPEG", quality=85)
    before = (tmp_path / "1.jpg").stat().st_size
    shrink_thumbnails(tmp_path, (216, 288), 80)
    assert (tmp_path / "1.jpg").stat().st_size < before


def test_shrink_leaves_no_temp_files_and_ignores_other_files(tmp_path):
    _write(tmp_path / "1.jpg", (288, 384))
    (tmp_path / ".9.jpg.tmp").write_bytes(b"half written")
    (tmp_path / "notes.txt").write_text("hi")
    assert shrink_thumbnails(tmp_path, (216, 288), 80) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".9.jpg.tmp",
        "1.jpg",
        "notes.txt",
    ]


def test_shrink_crops_to_the_target_aspect_ratio(tmp_path):
    _write(tmp_path / "1.jpg", (400, 200))  # landscape source
    shrink_thumbnails(tmp_path, (216, 288), 80)
    with Image.open(tmp_path / "1.jpg") as im:
        assert im.size == (216, 288)


def test_shrink_of_a_missing_directory_is_a_no_op(tmp_path):
    assert shrink_thumbnails(tmp_path / "nope", (216, 288), 80) == 0
