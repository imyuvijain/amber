"""FastAPI app: users, filters, the pair flow, choosing, results, thumbnails.

One sqlite connection is shared by every request, so every handler that touches it holds
`lock` (sync handlers run in a threadpool). If `web/dist` exists it is served at `/`.
"""

import json
import math
import re
import threading
from dataclasses import asdict
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from amber.config import REPO_ROOT
from amber.context import DOMAINS, AppContext
from amber.data.catalog import thumb_path
from amber.eval.metrics import learning_curve, score_showdown
from amber.flow import CatalogExhausted, next_pair, progress
from amber.store.db import (
    AlreadyAnswered,
    Choice,
    MissingPredictions,
    NotFound,
    answer_pair,
    create_user,
    get_predictions,
    list_choices,
    save_answer,
    user_exists,
)

WEB_DIST = REPO_ROOT / "web" / "dist"
DEV_ORIGIN = "http://localhost:5173"
_THUMB_NAME = re.compile(r"[0-9]+\.jpg")
_NOT_LABELS = ("id", "title")


class UserIn(BaseModel):
    name: str


class FiltersIn(BaseModel):
    genders: list[str]
    sections: list[str]


class DescribeIn(BaseModel):
    text: str


class NextIn(BaseModel):
    domain: Literal["clothes", "art"]
    phase: Literal["train", "showdown", "retest"]


class ChooseIn(BaseModel):
    chosen_id: str


def _label(value) -> str | None:
    """A catalog cell as a label string, or None when it is empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text or None


def _item_table(items: pd.DataFrame, domain: str) -> dict[str, dict]:
    """Every catalog row as an `Item` dict, by id."""
    label_cols = [c for c in items.columns if c not in _NOT_LABELS]
    table = {}
    for row in items.to_dict("records"):
        labels = {c: s for c in label_cols if (s := _label(row[c])) is not None}
        n = row["id"].split(":", 1)[1]
        table[row["id"]] = {
            "id": row["id"],
            "domain": domain,
            "title": row["title"],
            "thumb_url": f"/thumbs/{domain}/{n}.jpg",
            "labels": labels,
        }
    return table


def create_app(ctx: AppContext) -> FastAPI:
    app = FastAPI(title="Amber")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[DEV_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    lock = threading.Lock()
    item_tables = {d: _item_table(ctx.items[d], d) for d in DOMAINS}

    def require_user(user_id: str) -> None:
        if not user_exists(ctx.conn, user_id):
            raise HTTPException(404, f"no such user: {user_id}")

    @app.post("/api/users")
    def post_user(body: UserIn) -> dict:
        with lock:
            return {"user_id": create_user(ctx.conn, body.name)}

    @app.put("/api/users/{uid}/filters")
    def put_filters(uid: str, body: FiltersIn) -> dict:
        with lock:
            require_user(uid)
            clothes = ctx.items["clothes"]
            allowed = int(
                (
                    clothes["gender"].isin(body.genders)
                    & clothes["master_category"].isin(body.sections)
                ).sum()
            )
            if allowed == 0:
                raise HTTPException(422, "those filters leave no clothes to show")
            save_answer(ctx.conn, uid, "filters", "genders", json.dumps(body.genders))
            save_answer(ctx.conn, uid, "filters", "sections", json.dumps(body.sections))
            return {"allowed": allowed}

    @app.put("/api/users/{uid}/describe")
    def put_describe(uid: str, body: DescribeIn) -> dict:
        with lock:
            require_user(uid)
            save_answer(ctx.conn, uid, "describe", "text", body.text)
            return {}

    @app.post("/api/users/{uid}/next")
    def post_next(uid: str, body: NextIn) -> dict:
        with lock:
            require_user(uid)
            if body.phase == "showdown" and body.domain == "clothes":
                art = progress(ctx.conn, uid, "art", "train", ctx)
                if art["done"] < art["total"]:
                    raise HTTPException(409, "finish the art picks first")
            try:
                pair = next_pair(ctx.conn, uid, body.domain, body.phase, ctx)
            except CatalogExhausted as exc:
                raise HTTPException(409, f"catalog exhausted: {exc}") from exc
            except ValueError as exc:  # e.g. a showdown on art
                raise HTTPException(422, str(exc)) from exc
            except NotImplementedError as exc:
                raise HTTPException(501, str(exc)) from exc
            prog = progress(ctx.conn, uid, body.domain, body.phase, ctx)
            if pair is None:
                return {"done": True, "progress": prog}
            return {
                "pair_id": pair.pair_id,
                "a": item_tables[body.domain][pair.a_id],
                "b": item_tables[body.domain][pair.b_id],
                "progress": prog,
            }

    def reveal_for(pair: Choice) -> dict[str, bool] | None:
        if pair.phase != "showdown":
            return None
        chose_a = pair.chosen_id == pair.a_id
        return {
            you: (p_a > 0.5) == chose_a
            for you, p_a in get_predictions(ctx.conn, pair.pair_id).items()
            if p_a != 0.5  # an abstaining You has no side to be right or wrong on
        }

    @app.post("/api/pairs/{pair_id}/choose")
    def post_choose(pair_id: str, body: ChooseIn) -> dict:
        with lock:
            try:
                pair = answer_pair(ctx.conn, pair_id, body.chosen_id)
            except NotFound as exc:
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except (AlreadyAnswered, MissingPredictions) as exc:
                raise HTTPException(409, str(exc)) from exc
            return {
                "progress": progress(
                    ctx.conn, pair.user_id, pair.domain, pair.phase, ctx
                ),
                "reveal": reveal_for(pair),
            }

    @app.get("/api/users/{uid}/results")
    def get_results(uid: str) -> dict:
        with lock:
            require_user(uid)
            choices = list_choices(ctx.conn, uid, "clothes", "showdown")
            predictions = {
                c.pair_id: get_predictions(ctx.conn, c.pair_id) for c in choices
            }
            total = ctx.settings.clothes_pairs
            late = (total // 2 + 1, total)  # e.g. picks 21-40 of 40
            all_scores = score_showdown(choices, predictions)
            return {
                "scores_all": [asdict(s) for s in all_scores],
                "scores_late": [
                    asdict(s) for s in score_showdown(choices, predictions, window=late)
                ],
                "curves": {
                    s.you: learning_curve(choices, predictions, s.you)
                    for s in all_scores
                },
                "n_clothes": len(choices),
            }

    @app.get("/thumbs/{domain}/{name}")
    def get_thumb(domain: str, name: str) -> FileResponse:
        prefix = {"clothes": "c", "art": "a"}.get(domain)
        if prefix is None or not _THUMB_NAME.fullmatch(name):
            raise HTTPException(404, "no such thumbnail")
        path = thumb_path(f"{prefix}:{name[:-4]}", data_dir=ctx.data_dir)
        if not path.is_file():
            raise HTTPException(404, "no such thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    if WEB_DIST.is_dir():  # mounted last, so it never shadows the routes above
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app
