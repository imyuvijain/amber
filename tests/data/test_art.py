import io
import itertools
import json
import logging
import re
from collections import Counter

import httpx
import pandas as pd
import pytest
from PIL import Image

from amber.data import art
from amber.data.art import AIC_HEADERS, KEEP_ART_TYPES, ingest_art
from amber.data.catalog import load_items, thumb_path

COLUMNS = ["id", "title", "artist", "date", "artwork_type"]
SEARCH_PATH = "/api/v1/artworks/search"
IMAGE_PATH = re.compile(r"/iiif/2/(?P<image_id>[^/]+)/full/843,/0/default\.jpg")


def _jpeg(size, rgb=(180, 60, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, "JPEG")
    return buf.getvalue()


def _artwork(art_id, kind="Painting", **overrides):
    record = {
        "id": art_id,
        "title": f"Work {art_id}",
        "artist_title": f"Artist {art_id}",
        "date_display": f"18{art_id:02d}",
        "artwork_type_title": kind,
        "image_id": f"img-{art_id}",
    }
    record.update(overrides)
    return record


TYPE_FIELD = "artwork_type_title.keyword"


def searched_type(request):
    """The one artwork type a search request asks for (`term` on the `.keyword` field)."""
    filters = json.loads(request.content)["query"]["bool"]["filter"]
    (kind,) = [
        f["term"][TYPE_FIELD] for f in filters if TYPE_FIELD in f.get("term", {})
    ]
    return kind


class FakeAIC:
    """A stand-in for api.artic.edu + the IIIF image server behind `httpx.MockTransport`.

    A search is answered for the one artwork type it filters on: the `artworks` of that
    type, served 100 (or whatever `limit` the request asks for) at a time, in order.
    `images` maps image_id -> JPEG bytes and anything else is a 404. `faults` are
    callables `(request) -> Response | None` tried first, to script failures.
    """

    def __init__(self, artworks, images):
        self.artworks = artworks
        self.images = images
        self.requests: list[httpx.Request] = []
        self.faults = []
        self.window = 1000  # the live API refuses to page past this many results

    @property
    def searches(self):
        return [r for r in self.requests if r.url.path == SEARCH_PATH]

    def searches_for(self, kind):
        return [r for r in self.searches if searched_type(r) == kind]

    def pages_for(self, kind):
        return [json.loads(r.content)["page"] for r in self.searches_for(kind)]

    def pool(self, request):
        """What a search request is entitled to: artworks of the type it filters on."""
        kind = searched_type(request)
        return [a for a in self.artworks if a["artwork_type_title"] == kind]

    @property
    def image_requests(self):
        return [r for r in self.requests if r.url.path != SEARCH_PATH]

    def __call__(self, request):
        self.requests.append(request)
        for fault in self.faults:
            response = fault(request)
            if response is not None:
                return response
        if request.url.path == SEARCH_PATH:
            assert request.method == "POST" and request.url.host == "api.artic.edu"
            body = json.loads(request.content)
            page, limit = body["page"], body["limit"]
            if page * limit > self.window:  # what api.artic.edu answers (checked live)
                return httpx.Response(
                    403,
                    json={
                        "status": 403,
                        "error": "Invalid number of results",
                        "detail": "You have requested too many results.",
                    },
                )
            pool = self.pool(request)
            chunk = pool[(page - 1) * limit : page * limit]
            pages = -(-len(pool) // limit)
            pagination = {
                "total": len(pool),
                "limit": limit,
                "total_pages": pages,
                "current_page": page,
            }
            return httpx.Response(200, json={"pagination": pagination, "data": chunk})
        match = IMAGE_PATH.fullmatch(request.url.path)
        assert match and request.url.host == "www.artic.edu", request.url
        image = self.images.get(match["image_id"])
        if image is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200, content=image, headers={"Content-Type": "image/jpeg"}
        )

    def client(self):
        # no default headers: the code under test must add AIC_HEADERS itself
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def aic():
    """Three artworks; a:3's image 404s (it is not served)."""
    return FakeAIC(
        [_artwork(1), _artwork(2, "Print"), _artwork(3, "Photograph")],
        {"img-1": _jpeg((800, 600)), "img-2": _jpeg((300, 900))},
    )


@pytest.fixture
def mock_client(aic):
    return aic.client()


@pytest.fixture
def seen_requests(aic):
    return aic.requests


def _many(count, size=(400, 300), **kwargs):
    """`count` artworks (ids 1..count) with a servable image each."""
    artworks = [_artwork(i, **kwargs) for i in range(1, count + 1)]
    image = _jpeg(size)
    return FakeAIC(artworks, {f"img-{i}": image for i in range(1, count + 1)})


def _by_type(counts, size=(400, 300)):
    """`counts` maps artwork type -> how many artworks it has. Ids are 1000 * (the
    type's position in KEEP_ART_TYPES, from 1) + k: Painting 1001, ...; Photograph 2001, ..."""
    artworks = []
    for position, kind in enumerate(KEEP_ART_TYPES, start=1):
        artworks += [
            _artwork(1000 * position + k, kind)
            for k in range(1, counts.get(kind, 0) + 1)
        ]
    image = _jpeg(size)
    return FakeAIC(artworks, {a["image_id"]: image for a in artworks})


class FakeClock:
    """Replaces `art.time`: sleeping advances the clock instantly."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


# --- the plan's tests --------------------------------------------------------------


def test_ingest_art_filters_and_prefixes(tmp_path, mock_client):
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]  # a:3's image 404 -> skipped
    assert set(df.artwork_type) <= set(KEEP_ART_TYPES)
    assert max(Image.open(tmp_path / "thumbs" / "1.jpg").size) == 384


def test_ingest_art_sends_user_agent(tmp_path, mock_client, seen_requests):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert len(seen_requests) == 4 + 3  # a search per artwork type + three images
    assert all(r.headers["AIC-User-Agent"].startswith("amber") for r in seen_requests)


# --- the request shapes ------------------------------------------------------------


def test_constants():
    assert KEEP_ART_TYPES == (
        "Painting",
        "Photograph",
        "Print",
        "Drawing and Watercolor",
    )
    assert AIC_HEADERS == {
        "AIC-User-Agent": "amber (https://github.com/imyuvijain/amber)"
    }


def test_each_artwork_type_gets_its_own_search_with_the_same_filters(
    tmp_path, aic, mock_client
):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert [searched_type(r) for r in aic.searches] == list(KEEP_ART_TYPES)
    for search in aic.searches:
        assert str(search.url) == "https://api.artic.edu/api/v1/artworks/search"
        body = json.loads(search.content)
        assert body["limit"] == 100 and body["page"] == 1
        assert body["fields"] == [
            "id",
            "title",
            "artist_title",
            "date_display",
            "artwork_type_title",
            "image_id",
        ]
        # `.keyword`: the plain field is analysed text and matches nothing (checked live)
        assert body["query"]["bool"]["filter"] == [
            {"term": {"is_public_domain": True}},
            {"term": {TYPE_FIELD: searched_type(search)}},
            {"exists": {"field": "image_id"}},
        ]


def test_images_come_from_the_iiif_server_at_843_wide(tmp_path, aic, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert sorted(str(r.url) for r in aic.image_requests) == [
        f"https://www.artic.edu/iiif/2/img-{i}/full/843,/0/default.jpg"
        for i in (1, 2, 3)
    ]


# --- what gets written -------------------------------------------------------------


def test_parquet_has_expected_columns_and_values(tmp_path, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    df = pd.read_parquet(tmp_path / "items.parquet")
    assert list(df.columns) == COLUMNS
    first = df.set_index("id").loc["a:2"]
    assert first.title == "Work 2" and first.artist == "Artist 2"
    assert first.date == "1802" and first.artwork_type == "Print"


def test_catalog_is_readable_through_the_catalog_helpers(tmp_path, mock_client):
    out = tmp_path / "art"
    ingest_art(out, n=3, client=mock_client, min_interval=0)
    assert list(load_items("art", data_dir=tmp_path).id) == ["a:1", "a:2"]
    assert thumb_path("a:2", data_dir=tmp_path).exists()


def test_missing_artist_and_date_become_empty_strings(tmp_path):
    aic = FakeAIC(
        [_artwork(1, artist_title=None, date_display=None)], {"img-1": _jpeg((10, 10))}
    )
    df = ingest_art(tmp_path, n=1, client=aic.client(), min_interval=0)
    assert (df.artist[0], df.date[0]) == ("", "")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((800, 600), (384, 288)),  # landscape: width is the longest side
        ((300, 900), (128, 384)),  # portrait: height is the longest side
        ((600, 600), (384, 384)),
        ((100, 50), (384, 192)),  # a small source is scaled up to 384 as well
        ((843, 1), (384, 1)),  # extreme panorama never collapses to 0 px
    ],
)
def test_thumbnail_longest_side_is_384_with_aspect_kept(tmp_path, source, expected):
    aic = FakeAIC([_artwork(1)], {"img-1": _jpeg(source)})
    ingest_art(tmp_path, n=1, client=aic.client(), min_interval=0)
    with Image.open(tmp_path / "thumbs" / "1.jpg") as im:
        assert im.format == "JPEG" and im.size == expected


def test_no_temp_files_are_left_behind(tmp_path, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert sorted(p.name for p in (tmp_path / "thumbs").iterdir()) == ["1.jpg", "2.jpg"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["items.parquet", "thumbs"]


# --- skipping and failures ---------------------------------------------------------


def test_failed_image_is_logged_and_the_next_result_fills_in(
    tmp_path, aic, mock_client, caplog
):
    aic.artworks.append(_artwork(4))
    aic.images["img-4"] = _jpeg((400, 300))
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2", "a:4"]  # n counts saved thumbnails
    assert any("a:3" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "thumbs" / "3.jpg").exists()


def test_undecodable_image_bytes_are_skipped(tmp_path, aic, mock_client):
    aic.images["img-1"] = b"<html>definitely not an image</html>"
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:2"]
    assert not (tmp_path / "thumbs" / "1.jpg").exists()


def test_artwork_without_an_image_id_is_skipped_without_a_request(
    tmp_path, aic, mock_client
):
    aic.artworks[0]["image_id"] = None
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:2"]
    assert len(aic.image_requests) == 2  # img-2 and img-3 only


@pytest.mark.parametrize("n", [0, -5])
def test_n_must_be_positive(tmp_path, mock_client, n):
    with pytest.raises(ValueError):
        ingest_art(tmp_path, n=n, client=mock_client, min_interval=0)


# --- paging ------------------------------------------------------------------------


def test_pages_through_results_until_n_thumbnails_are_collected(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(7)
    df = ingest_art(tmp_path, n=5, client=aic.client(), min_interval=0)
    assert df.id.tolist() == [f"a:{i}" for i in range(1, 6)]
    assert aic.pages_for("Painting") == [1, 2, 3]  # stops at n: no page 4 fetched
    assert len(aic.image_requests) == 5  # and no sixth image either


def test_stops_when_the_results_run_out(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(4)  # exactly two full pages
    df = ingest_art(tmp_path, n=50, client=aic.client(), min_interval=0)
    assert len(df) == 4
    assert aic.pages_for("Painting") == [1, 2]


def test_stops_after_a_short_page_even_without_pagination_info(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(3)

    def bare_page(request):
        if request.url.path == SEARCH_PATH:
            page = json.loads(request.content)["page"]
            return httpx.Response(
                200, json={"data": aic.pool(request)[2 * page - 2 : 2 * page]}
            )

    aic.faults.append(bare_page)
    df = ingest_art(tmp_path, n=50, client=aic.client(), min_interval=0)
    assert len(df) == 3
    assert aic.pages_for("Painting") == [1, 2]


def test_never_asks_for_a_page_the_api_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "PAGE_SIZE", 1)
    monkeypatch.setattr(art, "MAX_RESULTS", 3)
    aic = _many(10)
    aic.window = 3  # a fourth page would be answered with a 403
    df = ingest_art(tmp_path, n=10, client=aic.client(), min_interval=0)
    assert len(df) == 3 and aic.pages_for("Painting") == [1, 2, 3]


def test_the_apis_real_1000_result_window_caps_a_type(tmp_path, caplog):
    # api.artic.edu answers 403 for page 11 at 100 per page: the first 1000 results are
    # all a search can reach, so a type that has 1500 contributes 1000, with a warning
    aic = _many(1500, size=(32, 24))
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        df = ingest_art(tmp_path, n=2000, client=aic.client(), min_interval=0)
    assert len(df) == 1000
    assert aic.pages_for("Painting") == list(range(1, 11))
    assert any(
        "1000" in r.getMessage() and "Painting" in r.getMessage()
        for r in caplog.records
    )


def test_each_types_window_is_separate(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    monkeypatch.setattr(art, "MAX_RESULTS", 4)
    aic = _by_type({kind: 6 for kind in KEEP_ART_TYPES})
    aic.window = 4
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        df = ingest_art(tmp_path, n=100, client=aic.client(), min_interval=0)
    assert Counter(df.artwork_type) == {kind: 4 for kind in KEEP_ART_TYPES}
    assert all(aic.pages_for(kind) == [1, 2] for kind in KEEP_ART_TYPES)
    # each type reports hitting its own window
    messages = [r.getMessage() for r in caplog.records]
    for kind in KEEP_ART_TYPES:
        assert sum(kind in m and "limit" in m for m in messages) == 1


def test_asking_for_more_than_all_the_windows_hold_warns_up_front(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        ingest_art(tmp_path, n=4001, client=_many(2).client(), min_interval=0)
    assert any(
        "4001" in r.getMessage() and "4000" in r.getMessage() for r in caplog.records
    )


def test_asking_for_2000_does_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        ingest_art(tmp_path, n=2000, client=_many(2).client(), min_interval=0)
    assert not caplog.records


# --- one search per artwork type, interleaved -----------------------------------------


def test_results_interleave_across_types(tmp_path):
    aic = _by_type({kind: 3 for kind in KEEP_ART_TYPES})
    df = ingest_art(tmp_path, n=8, client=aic.client(), min_interval=0)
    # one from each type in KEEP_ART_TYPES order, then the next from each, until n
    assert [r.url.path.split("/")[3] for r in aic.image_requests] == [
        f"img-{i}" for i in (1001, 2001, 3001, 4001, 1002, 2002, 3002, 4002)
    ]
    assert Counter(df.artwork_type) == {kind: 2 for kind in KEEP_ART_TYPES}


def test_an_exhausted_type_is_skipped_and_the_others_carry_on(tmp_path):
    aic = _by_type(
        {"Painting": 3, "Photograph": 1, "Print": 3, "Drawing and Watercolor": 0}
    )
    df = ingest_art(tmp_path, n=6, client=aic.client(), min_interval=0)
    assert [r.url.path.split("/")[3] for r in aic.image_requests] == [
        f"img-{i}" for i in (1001, 2001, 3001, 1002, 3002, 1003)
    ]
    assert len(df) == 6
    # a type that ran out is not searched again, and one with nothing is searched once
    assert aic.pages_for("Photograph") == [1]
    assert aic.pages_for("Drawing and Watercolor") == [1]


def test_types_are_only_searched_when_the_round_robin_reaches_them(tmp_path):
    aic = _by_type({kind: 3 for kind in KEEP_ART_TYPES})
    ingest_art(tmp_path, n=2, client=aic.client(), min_interval=0)
    assert [searched_type(r) for r in aic.searches] == ["Painting", "Photograph"]


def test_reaches_2000_with_a_balanced_mix(tmp_path):
    aic = _by_type({kind: 600 for kind in KEEP_ART_TYPES}, size=(32, 24))
    df = ingest_art(tmp_path, n=2000, client=aic.client(), min_interval=0)
    assert len(df) == 2000 and df.id.is_unique
    assert Counter(df.artwork_type) == {kind: 500 for kind in KEEP_ART_TYPES}
    assert all(aic.pages_for(kind) == [1, 2, 3, 4, 5] for kind in KEEP_ART_TYPES)


def test_a_type_with_few_artworks_contributes_fewer(tmp_path):
    aic = _by_type(
        {"Painting": 10, "Photograph": 2, "Print": 10, "Drawing and Watercolor": 10}
    )
    df = ingest_art(tmp_path, n=20, client=aic.client(), min_interval=0)
    assert len(df) == 20
    assert Counter(df.artwork_type) == {
        "Painting": 6,
        "Photograph": 2,
        "Print": 6,
        "Drawing and Watercolor": 6,
    }


def test_an_artwork_repeated_on_a_later_page_counts_once(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(4)
    aic.artworks.insert(2, aic.artworks[1])  # ranking shifted: a:2 shows up again
    df = ingest_art(tmp_path, n=4, client=aic.client(), min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2", "a:3", "a:4"]
    assert len(aic.image_requests) == 4  # a:2 is not downloaded twice


# --- what real artic.edu needs (found by running against it) ------------------------


def test_requests_use_an_identifying_user_agent_not_the_httpx_default(
    tmp_path, aic, mock_client
):
    # artic.edu's Cloudflare answers 403 to "python-httpx/x.y" even with AIC-User-Agent
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert aic.requests
    for request in aic.requests:
        assert request.headers["User-Agent"].startswith("amber")
        assert "httpx" not in request.headers["User-Agent"]
        assert request.headers["AIC-User-Agent"] == AIC_HEADERS["AIC-User-Agent"]


def test_aborts_when_images_keep_failing_instead_of_walking_every_result(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(art, "MAX_CONSECUTIVE_FAILURES", 3)
    aic = _many(10)
    # a:1 downloads fine, every later image is refused (as happened with the httpx UA)
    aic.faults.append(
        lambda r: (
            httpx.Response(403)
            if r.url.path != SEARCH_PATH and "/img-1/" not in r.url.path
            else None
        )
    )
    with pytest.raises(RuntimeError, match="3 images in a row"):
        ingest_art(tmp_path, n=5, client=aic.client(), min_interval=0)
    assert len(aic.image_requests) == 1 + 3  # and then it stopped (403 is not retried)
    assert list(pd.read_parquet(tmp_path / "items.parquet").id) == ["a:1"]


def test_a_success_resets_the_failure_streak(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "MAX_CONSECUTIVE_FAILURES", 3)
    aic = _many(6)
    for i in (1, 2, 4, 5):  # fail, fail, ok, fail, fail, ok: never 3 in a row
        del aic.images[f"img-{i}"]
    df = ingest_art(tmp_path, n=6, client=aic.client(), min_interval=0)
    assert df.id.tolist() == ["a:3", "a:6"]


# --- throttling --------------------------------------------------------------------


def test_requests_start_at_least_min_interval_apart(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(art, "time", clock)
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(5)
    starts = []
    aic.faults.append(lambda r: starts.append(clock.now))  # records, returns None
    ingest_art(tmp_path, n=5, client=aic.client(), min_interval=1.0)
    assert len(starts) == len(aic.requests) == 4 + 2 + 5  # 4 + 2 searches, 5 images
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert gaps == [pytest.approx(1.0)] * 10  # at least 1 s, and no longer than needed
    assert min(gaps) >= 1.0 - 1e-9


def test_throttle_counts_time_spent_working(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(art, "time", clock)
    aic = _many(2)
    # each request "takes" 0.4 s, so only 0.6 s of the 1 s interval is left to sleep
    aic.faults.append(lambda r: setattr(clock, "now", clock.now + 0.4))
    ingest_art(tmp_path, n=2, client=aic.client(), min_interval=1.0)
    assert len(aic.requests) == 4 + 2  # a search per type, two images
    assert clock.sleeps == [pytest.approx(0.6)] * 5


def test_no_sleeping_when_min_interval_is_zero(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(art, "time", clock)
    ingest_art(tmp_path, n=3, client=_many(3).client(), min_interval=0)
    assert clock.sleeps == []


# --- retries -----------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_search_errors_are_retried(tmp_path, aic, mock_client, status):
    left = [art.ATTEMPTS - 1]

    def flaky(request):
        if request.url.path == SEARCH_PATH and left[0]:
            left[0] -= 1
            return httpx.Response(status)

    aic.faults.append(flaky)
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    assert len(aic.searches) == art.ATTEMPTS + len(KEEP_ART_TYPES) - 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_image_errors_are_retried(tmp_path, aic, mock_client, status):
    left = [1]

    def flaky(request):
        if "img-1" in str(request.url) and left[0]:
            left[0] -= 1
            return httpx.Response(status)

    aic.faults.append(flaky)
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    assert sum("img-1" in str(r.url) for r in aic.image_requests) == 2


def test_connection_errors_are_retried(tmp_path, aic, mock_client):
    left = [1]

    def drop_once(request):
        if left[0]:
            left[0] -= 1
            raise httpx.ConnectError("connection reset")

    aic.faults.append(drop_once)
    assert len(ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)) == 2


def test_not_found_is_not_retried(tmp_path, aic, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert sum("img-3" in str(r.url) for r in aic.image_requests) == 1


def test_retries_back_off_instead_of_hammering_the_server(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(art, "time", clock)
    aic = _many(1)
    starts = []
    busy = [2]

    def too_many_requests(request):
        if request.url.path == SEARCH_PATH:
            starts.append(clock.now)
            if busy[0]:
                busy[0] -= 1
                return httpx.Response(429)

    aic.faults.append(too_many_requests)
    ingest_art(tmp_path, n=1, client=aic.client(), min_interval=1.0)
    first_wait, second_wait = starts[1] - starts[0], starts[2] - starts[1]
    assert first_wait > 1.0 and second_wait > first_wait


def test_search_error_that_persists_is_raised_and_keeps_collected_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(4)
    aic.faults.append(
        lambda r: (
            httpx.Response(503)
            if r.url.path == SEARCH_PATH and json.loads(r.content)["page"] == 2
            else None
        )
    )
    with pytest.raises(httpx.HTTPStatusError):
        ingest_art(tmp_path, n=4, client=aic.client(), min_interval=0)
    assert list(pd.read_parquet(tmp_path / "items.parquet").id) == ["a:1", "a:2"]
    # page 1 of each type, then Painting's page 2 until it gave up
    assert len(aic.searches) == len(KEEP_ART_TYPES) + art.ATTEMPTS


def test_image_server_error_that_persists_skips_that_image(tmp_path, aic, mock_client):
    aic.faults.append(lambda r: httpx.Response(500) if "img-1" in str(r.url) else None)
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:2"]


def test_network_failure_that_persists_stops_the_run(tmp_path, aic, mock_client):
    # a dead network must not turn into "skip every image" and an empty catalog
    def down(request):
        if request.url.path != SEARCH_PATH and "img-2" in str(request.url):
            raise httpx.ConnectError("network is unreachable")

    aic.faults.append(down)
    with pytest.raises(httpx.ConnectError):
        ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert list(pd.read_parquet(tmp_path / "items.parquet").id) == ["a:1"]


@pytest.mark.parametrize(
    "timeout", [httpx.ReadTimeout("timed out"), httpx.ConnectTimeout("timed out")]
)
def test_an_image_that_keeps_timing_out_is_skipped_not_fatal(tmp_path, caplog, timeout):
    # Result order is deterministic and a skipped image never gets a thumbnail, so if
    # one stuck image ended the run, every rerun would end at the same image.
    aic = _many(3)

    def stuck(request):
        if "/img-2/" in request.url.path:
            raise timeout

    aic.faults.append(stuck)
    with caplog.at_level(logging.WARNING, logger="amber.data.art"):
        df = ingest_art(tmp_path, n=3, client=aic.client(), min_interval=0)
    assert df.id.tolist() == ["a:1", "a:3"]
    assert sum("/img-2/" in r.url.path for r in aic.image_requests) == art.ATTEMPTS
    assert any("a:2" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "thumbs" / "2.jpg").exists()

    # and the next run is not stuck on it: it retries a:2 once more, the rest is resumed
    aic.faults.clear()
    aic.requests.clear()
    assert ingest_art(
        tmp_path, n=3, client=aic.client(), min_interval=0
    ).id.tolist() == [
        "a:1",
        "a:2",
        "a:3",
    ]
    assert [r.url.path for r in aic.image_requests] == [
        "/iiif/2/img-2/full/843,/0/default.jpg"
    ]


def test_image_timeouts_count_towards_the_failure_streak(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "MAX_CONSECUTIVE_FAILURES", 3)
    aic = _many(10)

    def everything_times_out(request):
        if request.url.path != SEARCH_PATH:
            raise httpx.ReadTimeout("timed out")

    aic.faults.append(everything_times_out)
    with pytest.raises(RuntimeError, match="3 images in a row"):
        ingest_art(tmp_path, n=10, client=aic.client(), min_interval=0)
    assert len(aic.image_requests) == 3 * art.ATTEMPTS  # three images, all attempts


def test_a_search_that_keeps_timing_out_is_still_fatal(tmp_path):
    aic = _many(3)

    def search_times_out(request):
        if request.url.path == SEARCH_PATH:
            raise httpx.ReadTimeout("timed out")

    aic.faults.append(search_times_out)
    with pytest.raises(httpx.ReadTimeout):
        ingest_art(tmp_path, n=3, client=aic.client(), min_interval=0)
    assert len(aic.searches) == art.ATTEMPTS


# --- resume ------------------------------------------------------------------------


def test_resume_does_not_download_existing_thumbnails_but_keeps_their_rows(
    tmp_path, aic, mock_client
):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    mtimes = {p: p.stat().st_mtime for p in (tmp_path / "thumbs").iterdir()}
    aic.requests.clear()
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    # only the broken a:3 is retried; a:1 and a:2 are not fetched again
    assert [str(r.url) for r in aic.image_requests] == [
        "https://www.artic.edu/iiif/2/img-3/full/843,/0/default.jpg"
    ]
    assert {p: p.stat().st_mtime for p in (tmp_path / "thumbs").iterdir()} == mtimes


def test_resumed_thumbnails_count_towards_n(tmp_path):
    aic = _many(3)
    ingest_art(tmp_path, n=1, client=aic.client(), min_interval=0)  # a:1 only
    assert [r.url.path for r in aic.image_requests] == [
        "/iiif/2/img-1/full/843,/0/default.jpg"
    ]
    aic.requests.clear()
    df = ingest_art(tmp_path, n=2, client=aic.client(), min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    assert [r.url.path for r in aic.image_requests] == [  # a:1 skipped, a:2 fetched
        "/iiif/2/img-2/full/843,/0/default.jpg"
    ]


def test_resume_recovers_rows_when_parquet_is_missing(tmp_path, aic, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    (tmp_path / "items.parquet").unlink()
    aic.requests.clear()
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    assert pd.read_parquet(tmp_path / "items.parquet").id.tolist() == ["a:1", "a:2"]


def test_merges_with_existing_parquet_without_duplicates(tmp_path, aic, mock_client):
    ingest_art(tmp_path, n=1, client=mock_client, min_interval=0)
    assert pd.read_parquet(tmp_path / "items.parquet").id.tolist() == ["a:1"]
    aic.artworks.reverse()  # the second run sees a:3, a:2, a:1
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"] and df.id.is_unique
    assert pd.read_parquet(tmp_path / "items.parquet").id.tolist() == ["a:1", "a:2"]


def test_new_metadata_replaces_the_old_row(tmp_path, aic, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    aic.artworks[0]["title"] = "Retitled"
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.set_index("id").loc["a:1"].title == "Retitled"
    assert len(df) == 2


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("network down"), KeyboardInterrupt()]
)
def test_interrupted_run_keeps_what_it_collected(tmp_path, monkeypatch, error):
    monkeypatch.setattr(art, "PAGE_SIZE", 2)
    aic = _many(5)

    def die_on_page_two(request):
        if request.url.path == SEARCH_PATH and json.loads(request.content)["page"] == 2:
            raise error

    aic.faults.append(die_on_page_two)
    with pytest.raises(type(error)):
        ingest_art(tmp_path, n=5, client=aic.client(), min_interval=0)
    assert list(pd.read_parquet(tmp_path / "items.parquet").id) == ["a:1", "a:2"]
    # the next run picks up where it stopped and does not fetch a:1, a:2 again
    aic.faults.clear()
    aic.requests.clear()
    assert len(ingest_art(tmp_path, n=5, client=aic.client(), min_interval=0)) == 5
    assert len(aic.image_requests) == 3


def test_interrupted_run_merges_with_existing_parquet(tmp_path):
    aic = _many(3)
    ingest_art(tmp_path, n=1, client=aic.client(), min_interval=0)  # a:1
    aic.artworks = aic.artworks[1:]  # the next run never sees a:1 again

    def die_on_third_image(request):
        if "img-3" in str(request.url):
            raise KeyboardInterrupt

    aic.faults.append(die_on_third_image)
    with pytest.raises(KeyboardInterrupt):
        ingest_art(tmp_path, n=3, client=aic.client(), min_interval=0)
    assert pd.read_parquet(tmp_path / "items.parquet").id.tolist() == ["a:1", "a:2"]


def test_failure_before_any_row_leaves_no_parquet(tmp_path, aic, mock_client):
    aic.faults.append(lambda r: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert not (tmp_path / "items.parquet").exists()


def test_no_results_writes_an_empty_catalog(tmp_path):
    df = ingest_art(tmp_path, n=5, client=FakeAIC([], {}).client(), min_interval=0)
    assert len(df) == 0 and list(df.columns) == COLUMNS
    assert len(pd.read_parquet(tmp_path / "items.parquet")) == 0


# --- the client --------------------------------------------------------------------


def test_creates_and_closes_its_own_client_when_none_is_given(
    tmp_path, aic, monkeypatch
):
    client, made = aic.client(), []

    def make_client(**kwargs):  # httpx.Client, replaced by the mocked one
        made.append(kwargs)
        return client

    monkeypatch.setattr(art.httpx, "Client", make_client)
    df = ingest_art(tmp_path, n=3, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]
    assert client.is_closed
    assert made[0].get("timeout")  # explicit; httpx's 5 s default is tight for IIIF
    assert all("AIC-User-Agent" in r.headers for r in aic.requests)


def test_does_not_close_a_client_it_was_given(tmp_path, mock_client):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert not mock_client.is_closed


@pytest.mark.slow
def test_real_api_first_artworks(tmp_path):
    df = ingest_art(tmp_path, n=4, min_interval=1.0)  # one round: a work of each type
    assert len(df) == 4 and df.id.str.startswith("a:").all()
    assert set(df.artwork_type) == set(KEEP_ART_TYPES)
    for item_id in df.id:
        with Image.open(tmp_path / "thumbs" / f"{item_id[2:]}.jpg") as im:
            assert max(im.size) == 384
