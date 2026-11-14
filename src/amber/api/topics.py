"""The veto screen's API: ranked topics with drivers, and veto / boost / undo taps.

Reads ``<data dir>/topics/topics.parquet`` (from ``amber topics build``) and keeps taps
in ``<data dir>/topics/feedback.json``. Local only: names never leave the machine.
"""

import threading
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from amber.topics.feedback import FeedbackStore, ranked_topics
from amber.topics.relevance import FEATURES


class Tap(BaseModel):
    action: Literal["veto", "boost", "clear"]
    shown: list[str] = []


def topics_router(data_dir: Path) -> APIRouter:
    router = APIRouter()
    lock = threading.Lock()
    topics_dir = Path(data_dir) / "topics"
    cache: dict = {}

    def table() -> pd.DataFrame:
        path = topics_dir / "topics.parquet"
        if not path.exists():
            raise HTTPException(404, "no topics yet: run `amber topics build`")
        mtime = path.stat().st_mtime
        if cache.get("mtime") != mtime:
            t = pd.read_parquet(path)
            t.index = t.index.astype(str)
            if "n_docs" not in t:
                t["n_docs"] = 0
            cache.update(mtime=mtime, table=t)
        return cache["table"]

    def store() -> FeedbackStore:
        return FeedbackStore(topics_dir / "feedback.json")

    @router.get("/api/topics")
    def get_topics() -> dict:
        t = table()
        with lock:
            return ranked_topics(t, t, store())

    @router.post("/api/topics/{topic_id}/feedback")
    def tap(topic_id: str, body: Tap) -> dict:
        t = table()
        if topic_id not in t.index:
            raise HTTPException(404, f"no such topic: {topic_id}")
        with lock:
            s = store()
            if body.action == "clear":
                s.clear(topic_id)
            else:
                s.record(topic_id, body.action, t.loc[topic_id, FEATURES].to_dict(),
                         shown=body.shown, features_of=t)
            return ranked_topics(t, t, s)

    return router
