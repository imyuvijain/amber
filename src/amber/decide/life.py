"""A person's life timeline, used as context for their decisions.

Behaviour depends on where you are and what season of life you're in: a Tokyo
night is a Waterloo morning, and board-exam weeks look nothing like a summer
break. ``load_timeline`` reads a small private YAML file (homes, life phases,
exam windows, birth year); ``annotate`` adds per-event context columns:

- ``country`` and ``tz``: home by date, overridden on days a country-specific
  Google Maps domain (``google.co.jp``, ``google.co.in``, ...) shows you abroad.
  Gaps of up to ``TRIP_FILL_DAYS`` between two sightings of the same country are
  filled. ``google.com`` says nothing about location and is ignored.
- ``local_hour``, ``local_weekday``, ``local_date``
- ``age`` in years (birth year only, so +/- 6 months)
- ``phase`` (or None), ``in_exam``, ``days_to_exam`` (to the next exam window's
  start, 0 inside one, capped at ``EXAM_HORIZON_DAYS``)
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

TRIP_FILL_DAYS = 7
EXAM_HORIZON_DAYS = 60
# Country-specific Google domains -> (country, timezone). One zone per country is a
# simplification (fine for JP/IN/TH; wrong across US or Canadian zones).
MAPS_DOMAINS: dict[str, tuple[str, str]] = {
    "co.jp": ("JP", "Asia/Tokyo"),
    "co.in": ("IN", "Asia/Kolkata"),
    "co.th": ("TH", "Asia/Bangkok"),
    "ca": ("CA", "America/Toronto"),
    "co.uk": ("GB", "Europe/London"),
    "de": ("DE", "Europe/Berlin"),
    "fr": ("FR", "Europe/Paris"),
    "com.sg": ("SG", "Asia/Singapore"),
    "co.kr": ("KR", "Asia/Seoul"),
    "com.au": ("AU", "Australia/Sydney"),
    "ae": ("AE", "Asia/Dubai"),
}
_MAPS = re.compile(r"https?://(?:www\.|maps\.)?google\.([a-z.]+)/maps")


@dataclass
class Timeline:
    birth_year: int | None
    homes: list[dict]  # sorted by "from"; each {from, tz, country}
    phases: list[dict] = field(default_factory=list)  # {from, to, name}
    exams: list[dict] = field(default_factory=list)  # {from, to, name}


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))


def load_timeline(path: Path) -> Timeline:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    norm = lambda items: sorted(
        ({**i, "from": _d(i["from"]), **({"to": _d(i["to"])} if "to" in i else {})} for i in items or []),
        key=lambda i: i["from"],
    )
    homes = norm(raw.get("homes"))
    if not homes:
        raise ValueError("timeline needs at least one home")
    return Timeline(raw.get("birth_year"), homes, norm(raw.get("phases")), norm(raw.get("exams")))


def _home(day: date, tl: Timeline) -> dict:
    current = tl.homes[0]
    for h in tl.homes:
        if h["from"] <= day:
            current = h
    return current


def daily_location(events: pd.DataFrame, tl: Timeline) -> pd.Series:
    """Country seen per UTC day from country-specific Maps URLs (sightings only)."""
    urls = events.get("url", pd.Series(dtype=object)).fillna("")
    dom = urls.str.extract(_MAPS)[0]
    hit = dom.map(lambda d: MAPS_DOMAINS.get(d, (None,))[0])
    seen = pd.DataFrame({"day": events["ts"].dt.tz_convert("UTC").dt.date, "cc": hit}).dropna()
    if seen.empty:
        return pd.Series(dtype=object)
    return seen.groupby("day").cc.agg(lambda s: s.value_counts().index[0]).sort_index()


def _country_for(day: date, sightings: pd.Series, tl: Timeline) -> tuple[str, str]:
    home = _home(day, tl)
    if not sightings.empty:
        days = sightings.index
        i = days.searchsorted(day)
        if i < len(days) and days[i] == day:
            cc = sightings.iloc[i]
        else:
            before = sightings.iloc[i - 1] if i > 0 else None
            after = sightings.iloc[i] if i < len(days) else None
            close = (
                before is not None and after is not None and before == after
                and (days[i] - days[i - 1]).days <= TRIP_FILL_DAYS
            )
            cc = before if close else None
        if cc and cc != home["country"]:
            tz = next((z for c, z in MAPS_DOMAINS.values() if c == cc), home["tz"])
            return cc, tz
    return home["country"], home["tz"]


def _window(day: date, items: list[dict]) -> dict | None:
    return next((i for i in items if i["from"] <= day <= i.get("to", i["from"])), None)


def _days_to_exam(day: date, exams: list[dict]) -> int:
    if _window(day, exams):
        return 0
    ahead = [(e["from"] - day).days for e in exams if e["from"] > day]
    return min(min(ahead), EXAM_HORIZON_DAYS) if ahead else EXAM_HORIZON_DAYS


def annotate(events: pd.DataFrame, tl: Timeline) -> pd.DataFrame:
    """Return a copy of ``events`` with the life-context columns described above."""
    out = events.copy()
    sightings = daily_location(out, tl)
    utc_day = out["ts"].dt.tz_convert("UTC").dt.date
    cache: dict[date, tuple[str, str]] = {}
    loc = [cache.setdefault(d, _country_for(d, sightings, tl)) for d in utc_day]
    out["country"] = [c for c, _ in loc]
    out["tz"] = [z for _, z in loc]
    local = [t.tz_convert(z) for t, z in zip(out["ts"], out["tz"])]
    out["local_hour"] = [t.hour for t in local]
    out["local_weekday"] = [t.weekday() for t in local]
    out["local_date"] = [t.date() for t in local]
    out["age"] = (
        [t.year + (t.dayofyear - 1) / 365.25 - (tl.birth_year + 0.5) for t in local]
        if tl.birth_year else float("nan")
    )
    out["phase"] = [(_window(d, tl.phases) or {}).get("name") for d in out["local_date"]]
    out["in_exam"] = [_window(d, tl.exams) is not None for d in out["local_date"]]
    out["days_to_exam"] = [_days_to_exam(d, tl.exams) for d in out["local_date"]]
    return out
