import pandas as pd
import pytest

from amber.decide.life import annotate, daily_location, load_timeline

YAML = """
birth_year: 2008
homes:
  - {from: 2008-01-01, tz: Asia/Tokyo, country: JP}
  - {from: 2026-09-02, tz: America/Toronto, country: CA}
phases:
  - {from: 2025-04-01, to: 2026-03-31, name: class 12}
  - {from: 2026-09-02, to: 2030-08-31, name: uni}
exams:
  - {from: 2026-02-17, to: 2026-04-04, name: boards}
  - {from: 2026-12-08, to: 2026-12-22, name: finals}
"""


@pytest.fixture
def tl(tmp_path):
    p = tmp_path / "life.yaml"
    p.write_text(YAML, encoding="utf-8")
    return load_timeline(p)


def ev(*rows):
    df = pd.DataFrame(rows, columns=["ts", "url"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["source"] = "google"
    df["kind"] = "activity"
    return df


def test_home_switches_on_move_date(tl):
    df = annotate(ev(("2026-08-01T03:00Z", None), ("2026-10-01T16:00Z", None)), tl)
    assert df["country"].tolist() == ["JP", "CA"]
    assert df["local_hour"].tolist() == [12, 12]  # 03 UTC = noon Tokyo; 16 UTC = noon EDT


def test_maps_trip_overrides_home_and_fills_short_gaps(tl):
    df = ev(
        ("2025-07-16T06:00Z", "https://www.google.co.in/maps/place/x"),
        ("2025-07-20T06:00Z", None),  # between two India sightings 6 days apart -> India
        ("2025-07-22T06:00Z", "https://www.google.co.in/maps/dir/y"),
        ("2025-08-20T06:00Z", None),  # far from any sighting -> home (Japan)
    )
    out = annotate(df, tl)
    assert out["country"].tolist() == ["IN", "IN", "IN", "JP"]
    assert out["local_hour"].iloc[0] == 11  # 06:00 UTC + 5:30, floored


def test_dotcom_maps_is_not_evidence(tl):
    loc = daily_location(ev(("2025-07-16T06:00Z", "https://www.google.com/maps/place/x")), tl)
    assert loc.empty


def test_phase_exam_and_age(tl):
    out = annotate(ev(("2026-03-01T00:00Z", None), ("2026-11-20T15:00Z", None), ("2024-01-01T00:00Z", None)), tl)
    assert out["phase"].iloc[:2].tolist() == ["class 12", "uni"]
    assert pd.isna(out["phase"].iloc[2])
    assert out["in_exam"].tolist() == [True, False, False]
    assert out["days_to_exam"].iloc[1] == pytest.approx(17, abs=1)  # finals start Dec 8
    assert out["days_to_exam"].iloc[0] == 0
    assert out["age"].iloc[1] == pytest.approx(18.4, abs=0.6)


def test_days_to_exam_capped_at_60(tl):
    out = annotate(ev(("2025-06-01T00:00Z", None)), tl)
    assert out["days_to_exam"].iloc[0] == 60
