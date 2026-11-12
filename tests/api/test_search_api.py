from datetime import datetime, timedelta

import pytest

from amber.flow import allowed_ids
from amber.yous.base import YOU_ORDER

MEN_APPAREL = {"genders": ["Men"], "sections": ["Apparel"]}


def _user(client, **filters) -> str:
    uid = client.post("/api/users", json={"name": "y"}).json()["user_id"]
    if filters:
        assert client.put(f"/api/users/{uid}/filters", json=filters).status_code == 200
    return uid


def _play(client, uid, domain, phase, choose="a", other=False) -> None:
    while not (
        r := client.post(
            f"/api/users/{uid}/next", json={"domain": domain, "phase": phase}
        ).json()
    ).get("done"):
        pick = r[choose]["id"]
        if other:
            pick = r["b" if r["a"]["id"] == pick else "a"]["id"]
        client.post(f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": pick})


def _ready(client) -> str:
    """Art finished and the showdown played: every You has something to say."""
    uid = _user(client, **MEN_APPAREL)
    client.put(f"/api/users/{uid}/style", json={"words": ["minimal"]})
    _play(client, uid, "art", "train")
    _play(client, uid, "clothes", "showdown")
    return uid


def _ids(items) -> list[str]:
    return [i["id"] for i in items]


def test_search_returns_items_for_both_columns(client, ctx):
    uid = _ready(client)
    res = client.get(f"/api/users/{uid}/search", params={"q": "red jacket"})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"you", "generic", "personal"}
    assert body["you"] == "art"  # the default
    for item in body["generic"] + body["personal"]:
        assert {"id", "domain", "title", "thumb_url", "labels"} <= set(item)
        assert item["domain"] == "clothes"
    assert len(body["generic"]) > 0 and len(body["personal"]) > 0


def test_search_never_returns_items_outside_the_filters(client, ctx):
    uid = _ready(client)
    allowed = allowed_ids(ctx.conn, uid, ctx)
    assert len(allowed) < len(ctx.items["clothes"])
    for you in YOU_ORDER:
        body = client.get(
            f"/api/users/{uid}/search", params={"q": "blue shirt", "you": you}
        ).json()
        assert set(_ids(body["generic"])) <= allowed
        assert set(_ids(body["personal"])) <= allowed


def test_each_you_is_accepted_and_abstainers_get_generic_only(client):
    uid = _user(
        client, **MEN_APPAREL
    )  # nothing answered: stated, bigfive, revealed abstain
    for you in YOU_ORDER:
        res = client.get(f"/api/users/{uid}/search", params={"q": "jacket", "you": you})
        assert res.status_code == 200
        assert res.json()["you"] == you
    body = client.get(
        f"/api/users/{uid}/search", params={"q": "jacket", "you": "revealed"}
    ).json()
    assert body["generic"] and body["personal"] == []


def test_the_chosen_you_changes_the_personal_column(client):
    uid = _ready(client)
    by_you = {
        you: client.get(
            f"/api/users/{uid}/search", params={"q": "shirt", "you": you}
        ).json()
        for you in ("art", "revealed")
    }
    assert by_you["art"]["generic"] == by_you["revealed"]["generic"]
    assert by_you["art"]["personal"] != by_you["revealed"]["personal"]


def test_unknown_you_is_422(client):
    uid = _user(client, **MEN_APPAREL)
    res = client.get(f"/api/users/{uid}/search", params={"q": "jacket", "you": "vibes"})
    assert res.status_code == 422


@pytest.mark.parametrize("q", ["", "   "])
def test_empty_query_is_422(client, q):
    uid = _user(client, **MEN_APPAREL)
    assert client.get(f"/api/users/{uid}/search", params={"q": q}).status_code == 422


def test_missing_query_is_422(client):
    uid = _user(client, **MEN_APPAREL)
    assert client.get(f"/api/users/{uid}/search").status_code == 422


def test_unknown_user_is_404(client):
    assert (
        client.get("/api/users/nope/search", params={"q": "jacket"}).status_code == 404
    )


def test_retest_picks_never_feed_a_you(client, ctx, monkeypatch):
    uid = _ready(client)
    before = client.get(
        f"/api/users/{uid}/search", params={"q": "shirt", "you": "revealed"}
    ).json()
    assert before["personal"]  # revealed has 6 picks to go on
    last = ctx.conn.execute(
        "SELECT MAX(answered_at) FROM pairs WHERE user_id = ? AND phase = 'showdown'",
        (uid,),
    ).fetchone()[0]
    monkeypatch.setenv(
        "AMBER_NOW", (datetime.fromisoformat(last) + timedelta(days=8)).isoformat()
    )
    _play(client, uid, "clothes", "retest", other=True)  # flips every earlier pick
    after = client.get(
        f"/api/users/{uid}/search", params={"q": "shirt", "you": "revealed"}
    ).json()
    assert after == before
