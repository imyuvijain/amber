"""Every builder, by You name, and a runner that keeps one failure from sinking the rest."""

import logging
import sqlite3
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from amber.yous.art_taste import build_art
from amber.yous.base import You, YouName
from amber.yous.bigfive import build_bigfive
from amber.yous.revealed import build_revealed
from amber.yous.stated import build_stated

if TYPE_CHECKING:
    from amber.context import AppContext

log = logging.getLogger(__name__)

Builder = Callable[[sqlite3.Connection, str, "AppContext"], You]

BUILDERS: dict[YouName, Builder] = {
    "stated": build_stated,
    "bigfive": build_bigfive,
    "art": build_art,
    "revealed": build_revealed,
}


# Built and frozen before the first clothing pair; none of them may see clothing choices.
FROZEN_YOUS: tuple[YouName, ...] = ("stated", "bigfive", "art")


def _run(
    names: Iterable[YouName], conn: sqlite3.Connection, user_id: str, ctx: "AppContext"
) -> dict[YouName, You]:
    """Run the registered builders among `names`. A builder that raises becomes an
    abstaining You whose meta is `{"error": "<ExcType>: <message>"}`."""
    yous: dict[YouName, You] = {}
    for name in names:
        if name not in BUILDERS:
            continue
        builder = BUILDERS[name]
        try:
            yous[name] = builder(conn, user_id, ctx)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            log.warning("builder %s failed: %s", name, exc)
            yous[name] = You(name, None, {"error": f"{type(exc).__name__}: {exc}"})
    return yous


def build_frozen_yous(
    conn: sqlite3.Connection, user_id: str, ctx: "AppContext"
) -> dict[YouName, You]:
    """The Yous frozen before the first clothing pair: `FROZEN_YOUS` that have a builder
    (all three, now that `bigfive` is registered). Failures isolate as in `build_all_yous`."""
    return _run(FROZEN_YOUS, conn, user_id, ctx)


def build_all_yous(
    conn: sqlite3.Connection, user_id: str, ctx: "AppContext"
) -> dict[YouName, You]:
    """Run every registered builder (frozen ones and revealed), for the readout."""
    return _run(tuple(BUILDERS), conn, user_id, ctx)
