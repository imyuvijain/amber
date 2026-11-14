import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from amber.api.topics import topics_router
from amber.topics.fake import fake_topic_docs
from amber.topics.relevance import score, topic_features


@pytest.fixture
def built(tmp_path):
    feats = topic_features(fake_topic_docs("planted"))
    feats["name"] = [f"name of {t}" for t in feats.index]
    feats["relevance"] = score(feats)
    (tmp_path / "topics").mkdir()
    feats.to_parquet(tmp_path / "topics" / "topics.parquet")
    return tmp_path


def _client(data_dir):
    app = FastAPI()
    app.include_router(topics_router(data_dir))
    return TestClient(app)


def test_topics_404_before_build(tmp_path):
    r = _client(tmp_path).get("/api/topics")
    assert r.status_code == 404 and "topics build" in r.json()["detail"]


def test_topics_ranked_with_drivers(built):
    r = _client(built).get("/api/topics")
    assert r.status_code == 200
    body = r.json()
    assert body["topics"][0]["id"] == "core"
    assert body["topics"][0]["drivers"]
    assert body["muted"] == [] and body["n_labels"] == 0


def test_veto_mutes_and_tunes_then_undo(built):
    c = _client(built)
    shown = [t["id"] for t in c.get("/api/topics").json()["topics"]]
    r = c.post("/api/topics/exam/feedback", json={"action": "veto", "shown": shown})
    assert r.status_code == 200
    body = r.json()
    assert "exam" not in [t["id"] for t in body["topics"]]
    assert [m["id"] for m in body["muted"]] == ["exam"]
    assert body["n_labels"] >= 1
    assert (built / "topics" / "feedback.json").exists()
    body = c.post("/api/topics/exam/feedback", json={"action": "clear"}).json()
    assert "exam" in [t["id"] for t in body["topics"]]


def test_boost_goes_first(built):
    body = _client(built).post("/api/topics/old/feedback", json={"action": "boost"}).json()
    assert body["topics"][0]["id"] == "old" and body["topics"][0]["state"] == "boost"


def test_bad_action_and_unknown_topic(built):
    c = _client(built)
    assert c.post("/api/topics/core/feedback", json={"action": "like"}).status_code == 422
    assert c.post("/api/topics/nope/feedback", json={"action": "veto"}).status_code == 404
