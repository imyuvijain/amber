"""Fetch public-domain artworks from the Art Institute of Chicago into thumbnails +
`items.parquet`.

One search per artwork type is paged through (100 per page, most prominent works first)
and the results are taken round-robin across the types, until `n` artworks have a saved
thumbnail. A single search reaches only its first 1000 results and the most prominent
ones are nearly all paintings, so one query per type gives both a larger reachable pool
and a balanced mix. Every request, search or image, is spaced at least
`min_interval` seconds from the one before and carries `AIC_HEADERS`, as the AIC asks of
API users. The run is resumable like the clothes ingest: an artwork whose thumbnail
already exists is not downloaded again, but its metadata is still collected.
"""

import io
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from PIL import Image

from amber.data.files import clean_text, save_thumbnail, write_catalog

logger = logging.getLogger(__name__)

KEEP_ART_TYPES = ("Painting", "Photograph", "Print", "Drawing and Watercolor")
AIC_HEADERS = {"AIC-User-Agent": "amber (https://github.com/imyuvijain/amber)"}
# www.artic.edu's Cloudflare answers 403 to the default `python-httpx/x.y` User-Agent (and
# to an empty one) even with AIC_HEADERS; an identifying UA plus AIC_HEADERS passes
USER_AGENT = "amber/0.1 (https://github.com/imyuvijain/amber)"

SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
IMAGE_URL = "https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"
FIELDS = [
    "id",
    "title",
    "artist_title",
    "date_display",
    "artwork_type_title",
    "image_id",
]
PAGE_SIZE = 100
# A search reaches only its first 1000 results: page 11 at 100 per page, or page 21 at 50,
# is answered 403 "Invalid number of results" (checked against the live API). Each
# artwork type is its own search, so each has its own window.
MAX_RESULTS = 1000
ART_THUMB_SIDE = 384  # longest side in pixels; the aspect ratio is kept

REQUEST_TIMEOUT = 30.0  # seconds; the IIIF server can be slow, httpx's default is 5
MAX_CONSECUTIVE_FAILURES = 25  # unusable images in a row before the run is aborted
ATTEMPTS = 3  # per request, for 429 / 5xx / connection errors
RETRY_BACKOFF = 5  # before attempt k+1 wait min_interval * RETRY_BACKOFF * k

COLUMNS = ["id", "title", "artist", "date", "artwork_type"]
_HEADERS = {**AIC_HEADERS, "User-Agent": USER_AGENT}


def _search_body(art_type: str, page: int) -> dict:
    return {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"is_public_domain": True}},
                    # `.keyword`: the plain field is analysed text, so matching on it
                    # finds nothing (checked against the live API)
                    {"term": {"artwork_type_title.keyword": art_type}},
                    {"exists": {"field": "image_id"}},
                ]
            }
        },
        "fields": FIELDS,
        "limit": PAGE_SIZE,
        "page": page,
    }


class _Fetcher:
    """Sends requests with `AIC_HEADERS` + `USER_AGENT`, `min_interval` apart, retrying."""

    def __init__(self, client: httpx.Client, min_interval: float) -> None:
        self.client = client
        self.min_interval = min_interval
        self._last_start: float | None = None

    def _pause(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _throttle(self) -> None:
        if self._last_start is not None:
            self._pause(self._last_start + self.min_interval - time.monotonic())
        self._last_start = time.monotonic()

    def send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """The response, however it came out: after `ATTEMPTS` tries the last response is
        returned (callers check its status), the last connection error is raised."""
        for attempt in range(1, ATTEMPTS + 1):
            self._throttle()
            try:
                response = self.client.request(method, url, headers=_HEADERS, **kwargs)
            except httpx.TransportError:
                if attempt == ATTEMPTS:
                    raise
            else:
                busy = response.status_code == 429 or response.status_code >= 500
                if not busy or attempt == ATTEMPTS:
                    return response
            self._pause(self.min_interval * RETRY_BACKOFF * attempt)
        raise AssertionError("unreachable")  # pragma: no cover


def _type_results(fetcher: _Fetcher, art_type: str) -> Iterator[dict]:
    """Artworks of one type from successive search pages, fetching a page only when asked
    for more than the pages so far held."""
    for page in range(1, MAX_RESULTS // PAGE_SIZE + 1):
        response = fetcher.send("POST", SEARCH_URL, json=_search_body(art_type, page))
        response.raise_for_status()
        payload = response.json()
        results = payload.get("data") or []
        yield from results
        total_pages = (payload.get("pagination") or {}).get("total_pages")
        if len(results) < PAGE_SIZE or (
            total_pages is not None and page >= total_pages
        ):
            return
    logger.warning(
        "stopped at the API's limit of %d results for %s", MAX_RESULTS, art_type
    )


def _search_results(fetcher: _Fetcher, art_types: Sequence[str]) -> Iterator[dict]:
    """One artwork from each type in turn, then the next from each, and so on; a type
    drops out once it has no more. A type is first searched when its turn first comes."""
    streams = [_type_results(fetcher, art_type) for art_type in art_types]
    while streams:
        for stream in list(streams):
            artwork = next(stream, None)
            if artwork is None:
                streams.remove(stream)
            else:
                yield artwork


def _thumbnail(data: bytes) -> Image.Image:
    """Decode and scale so the longest side is `ART_THUMB_SIDE`, keeping the aspect."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
    scale = ART_THUMB_SIDE / max(rgb.size)
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)


def _download_thumbnail(fetcher: _Fetcher, image_id: str) -> Image.Image:
    response = fetcher.send("GET", IMAGE_URL.format(image_id=image_id))
    response.raise_for_status()
    return _thumbnail(response.content)


def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=COLUMNS).astype("str")


def ingest_art(
    out_dir: Path,
    n: int = 2000,
    client: httpx.Client | None = None,
    min_interval: float = 1.0,
) -> pd.DataFrame:
    """Build `out_dir/items.parquet` and `out_dir/thumbs/<aic_id>.jpg`; return the table.

    Collects `n` public-domain artworks of the `KEEP_ART_TYPES` that have a saved
    thumbnail. Each type is searched separately (the API serves only the first
    `MAX_RESULTS` of a search) and taken in turns, so the mix is as even as the pools
    allow and a type with fewer artworks just contributes fewer.

    An image that fails (404, server error, timeout, undecodable) is logged and skipped
    and the next result takes its place. The run stops early when the results run out,
    and aborts with a RuntimeError after `MAX_CONSECUTIVE_FAILURES` unusable images in a
    row (something systematic is wrong). A refused or reset connection stops it at once.

    `client` is an `httpx.Client` to send through (tests pass a mock one); by default one
    is created and closed here. The returned/written table also keeps rows from a
    previous run's parquet, de-duplicated by `id`. If the run is interrupted (a network
    error, Ctrl-C, the abort above), the rows collected so far are saved before the
    exception propagates.
    """
    if n < 1:
        raise ValueError("n must be a positive number of artworks")
    reachable = len(KEEP_ART_TYPES) * MAX_RESULTS
    if n > reachable:
        logger.warning(
            "n=%d is more than the %d results the %d per-type searches can page "
            "through; collecting at most that many",
            n,
            reachable,
            len(KEEP_ART_TYPES),
        )
    out_dir = Path(out_dir)
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "items.parquet"
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True)
    fetcher = _Fetcher(client, min_interval)

    records: list[dict] = []
    seen: set[int] = set()
    skipped = 0
    failures_in_a_row = 0
    try:
        for artwork in _search_results(fetcher, KEEP_ART_TYPES):
            try:
                art_id = int(artwork["id"])
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "skipping result without a usable id: %r", artwork.get("id")
                )
                skipped += 1
                continue
            if art_id in seen:  # ranking shifted between pages
                continue
            seen.add(art_id)
            image_id = clean_text(artwork.get("image_id"))
            if not image_id:
                logger.warning("skipping a:%d, no image_id", art_id)
                skipped += 1
                continue
            dest = thumbs / f"{art_id}.jpg"
            if not dest.exists():
                try:
                    thumb = _download_thumbnail(fetcher, image_id)
                except (
                    Exception
                ) as exc:  # HTTP status, timeout, PIL error: skip the item
                    if isinstance(exc, httpx.TransportError) and not isinstance(
                        exc, httpx.TimeoutException
                    ):
                        # refused, reset, no route: the network is down. Stop (the rows
                        # so far are saved) instead of skipping every image after this.
                        # A timeout is skipped: results come in a fixed order and a
                        # skipped image has no thumbnail, so a rerun that died here
                        # would die at the same image every time.
                        raise
                    logger.warning("skipping a:%d, image not usable: %s", art_id, exc)
                    skipped += 1
                    failures_in_a_row += 1
                    if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                        # e.g. every image answers 403: don't walk every result
                        raise RuntimeError(
                            f"{failures_in_a_row} images in a row could not be fetched; "
                            f"last: {exc}"
                        ) from exc
                    continue
                failures_in_a_row = 0
                save_thumbnail(thumb, dest)
            records.append(
                {
                    "id": f"a:{art_id}",
                    "title": clean_text(artwork.get("title")),
                    "artist": clean_text(artwork.get("artist_title")),
                    "date": clean_text(artwork.get("date_display")),
                    "artwork_type": clean_text(artwork.get("artwork_type_title")),
                }
            )
            if len(records) % PAGE_SIZE == 0:
                logger.info("%d artworks collected", len(records))
            if len(records) >= n:
                break
    except BaseException:
        # network error, full disk, Ctrl-C: keep the metadata of the thumbnails written
        # so far (the next run would rebuild it anyway, but only after re-fetching)
        if records:
            items = write_catalog(parquet, thumbs, _frame(records))
            logger.warning("interrupted; saved %d artworks to the catalog", len(items))
        raise
    finally:
        if own_client:
            client.close()

    items = write_catalog(parquet, thumbs, _frame(records))
    logger.info("%d artworks in catalog (%d skipped this run)", len(items), skipped)
    return items
