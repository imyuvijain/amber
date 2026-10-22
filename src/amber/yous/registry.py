"""Every builder, by You name, and a runner that keeps one failure from sinking the rest."""

import logging
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING

from amber.yous.base import You, YouName
from amber.yous.stated import build_stated

if TYPE_CHECKING:
    from amber.context import AppContext

log = logging.getLogger(__name__)

Builder = Callable[[sqlite3.Connection, str, "AppContext"], You]

BUILDERS: dict[YouName, Builder] = {"stated": build_stated}


def build_all_yous(
    conn: sqlite3.Connection, user_id: str, ctx: "AppContext"
) -> dict[YouName, You]:
    """Run every registered builder. A builder that raises becomes an abstaining You whose
    meta is `{"error": "<ExcType>: <message>"}`."""
    yous: dict[YouName, You] = {}
    for name, builder in BUILDERS.items():
        try:
            yous[name] = builder(conn, user_id, ctx)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            log.warning("builder %s failed: %s", name, exc)
            yous[name] = You(name, None, {"error": f"{type(exc).__name__}: {exc}"})
    return yous
