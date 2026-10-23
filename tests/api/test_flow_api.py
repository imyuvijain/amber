import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from amber.api import app as app_module
from amber.api.app import create_app
from amber.flow import allowed_ids
from amber.store.db import get_predictions

MEN_APPAREL = {"genders": ["Men"], "sections": ["Apparel"]}


def _new_user(client, **filters) -> str:
    uid = client.post("/api/users", json={"name": "y"}).json()["user_id"]
    if filters:
        assert client.put(f"/api/users/{uid}/filters", json=filters).status_code == 200
    return uid


def _next(client, uid, domain="clothes", phase=None):
    phase = phase or ("train" if domain == "art" else "showdown")
    return client.post(
        f"/api/users/{uid}/next", json={"domain": domain, "phase": phase}
    )


def _ready(client, **filters) -> str:
    """A new user who has finished the art picks, so clothes can start."""
    uid = _new_user(client, **filters)
    _play(client, uid, "train", domain="art")
    return uid


def _play(client, uid, phase, choose="a", domain="clothes") -> list[dict]:
    """Answer every pair of a phase; returns the served `next` payloads."""
    served = []
    while not (r := _next(client, uid, domain, phase).json()).get("done"):
        served.append(r)
        resp = client.post(
            f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r[choose]["id"]}
        )
        assert resp.status_code == 200, resp.text
    return served


# --- the whole flow --------------------------------------------------------------


def test_full_flow_api(client, ctx):
    uid = client.post("/api/users", json={"name": "y"}).json()["user_id"]
    client.put(f"/api/users/{uid}/filters", json=MEN_APPAREL)
    client.put(f"/api/users/{uid}/style", json={"words": ["minimal"]})
    assert len(_play(client, uid, "train", domain="art")) == 6
    assert len(_play(client, uid, "showdown")) == 6
    res = client.get(f"/api/users/{uid}/results").json()
    assert res["n_clothes"] == 6
    assert {s["you"] for s in res["scores_all"]} >= {"stated", "art", "revealed"}
    for s in res["scores_all"]:
        assert set(s) == {"you", "correct", "n", "accuracy", "ci_low", "ci_high"}
        assert s["n"] == 6 and s["ci_low"] <= s["accuracy"] <= s["ci_high"]


def test_create_user(client, ctx):
    r = client.post("/api/users", json={"name": "Ada"})
    assert r.status_code == 200
    assert ctx.conn.execute(
        "SELECT name FROM users WHERE user_id = ?", (r.json()["user_id"],)
    ).fetchone() == ("Ada",)


def test_next_shape_and_progress(client):
    uid = _ready(client, **MEN_APPAREL)
    r = _next(client, uid).json()
    assert set(r) == {"pair_id", "a", "b", "progress"}
    assert r["progress"] == {"done": 0, "total": 6}
    for key in ("a", "b"):
        assert set(r[key]) == {"id", "domain", "title", "thumb_url", "labels"}
        assert r[key]["domain"] == "clothes"
    done = client.post(
        f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r["a"]["id"]}
    ).json()
    assert done["progress"] == {"done": 1, "total": 6}
    assert set(done["reveal"]) <= {"stated", "bigfive", "art", "revealed"}


def test_item_labels_are_non_empty_catalog_columns_as_strings(client, ctx):
    uid = _ready(client, **MEN_APPAREL)
    item = _next(client, uid).json()["a"]
    n = item["id"].split(":")[1]
    assert item["thumb_url"] == f"/thumbs/clothes/{n}.jpg"
    row = ctx.items["clothes"].set_index("id").loc[item["id"]]
    assert item["title"] == row["title"]
    assert set(item["labels"]) == {
        "gender",
        "master_category",
        "sub_category",
        "article_type",
        "base_colour",
        "season",
        "usage",
        "year",
    }
    assert item["labels"]["gender"] == "Men"
    assert item["labels"]["year"] == str(int(row["year"]))
    assert all(isinstance(v, str) and v for v in item["labels"].values())


def test_art_items_have_art_labels(client):
    uid = _new_user(client)
    item = _next(client, uid, "art").json()["a"]
    assert item["domain"] == "art"
    assert set(item["labels"]) == {"artist", "date", "artwork_type"}
    assert item["thumb_url"].startswith("/thumbs/art/")


def test_next_resumes_the_open_pair(client):
    uid = _ready(client, **MEN_APPAREL)
    first = _next(client, uid).json()
    assert _next(client, uid).json()["pair_id"] == first["pair_id"]


def test_clothes_phase_ends_with_done(client):
    uid = _ready(client, **MEN_APPAREL)
    assert len(_play(client, uid, "showdown")) == 6
    assert _next(client, uid).json() == {
        "done": True,
        "progress": {"done": 6, "total": 6},
    }


# --- filters, thumbnails ---------------------------------------------------------


def test_pairs_respect_filters(client, ctx):
    uid = _ready(client, **MEN_APPAREL)
    served = _play(client, uid, "showdown")
    assert len(served) == 6
    for pair in served:
        for item in (pair["a"], pair["b"]):
            assert item["labels"]["gender"] == "Men"
            assert item["labels"]["master_category"] == "Apparel"
    assert {i for p in served for i in (p["a"]["id"], p["b"]["id"])} <= allowed_ids(
        ctx.conn, uid, ctx
    )


def test_filters_return_the_allowed_count(client, ctx):
    uid = _new_user(client)
    r = client.put(f"/api/users/{uid}/filters", json=MEN_APPAREL)
    assert r.json() == {"allowed": 24}
    assert len(allowed_ids(ctx.conn, uid, ctx)) == 24
    r = client.put(
        f"/api/users/{uid}/filters",
        json={"genders": ["Men", "Women"], "sections": ["Apparel", "Footwear"]},
    )
    assert r.json() == {"allowed": 96}


def test_filters_allowing_nothing_are_rejected_and_not_saved(client, ctx):
    uid = _new_user(client, **MEN_APPAREL)
    for bad in (
        {"genders": [], "sections": ["Apparel"]},
        {"genders": ["Men"], "sections": []},
        {"genders": ["Nobody"], "sections": ["Apparel"]},
    ):
        assert client.put(f"/api/users/{uid}/filters", json=bad).status_code == 422
    assert len(allowed_ids(ctx.conn, uid, ctx)) == 24  # the old filters still stand


def test_every_served_item_has_thumbnail(client):
    uid = _ready(client, **MEN_APPAREL)
    served = _play(client, uid, "showdown")
    served += _play(client, _new_user(client), "train", domain="art")
    assert served
    for pair in served:
        for item in (pair["a"], pair["b"]):
            r = client.get(item["thumb_url"])
            assert r.status_code == 200, item["thumb_url"]
            assert r.headers["content-type"] == "image/jpeg"
            assert r.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    "path",
    [
        "/thumbs/clothes/9999.jpg",
        "/thumbs/clothes/abc.jpg",
        "/thumbs/clothes/1.png",
        "/thumbs/clothes/1.jpg.bak",
        "/thumbs/clothes/..%2F..%2Famber.db",
        "/thumbs/clothes/%2e%2e%2fitems.parquet",
        "/thumbs/shoes/1.jpg",
    ],
)
def test_thumbs_404_for_anything_else(client, path):
    assert client.get(path).status_code == 404


# --- choosing --------------------------------------------------------------------


def test_double_submit_is_safe(client):
    uid = _ready(client, **MEN_APPAREL)
    r = _next(client, uid).json()
    a, b = r["a"]["id"], r["b"]["id"]
    url = f"/api/pairs/{r['pair_id']}/choose"
    first = client.post(url, json={"chosen_id": a})
    again = client.post(url, json={"chosen_id": a})
    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()
    assert client.post(url, json={"chosen_id": b}).status_code == 409


def test_choose_errors(client):
    uid = _ready(client, **MEN_APPAREL)
    r = _next(client, uid).json()
    url = f"/api/pairs/{r['pair_id']}/choose"
    assert client.post(url, json={"chosen_id": "c:99999"}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert (
        client.post("/api/pairs/nope/choose", json={"chosen_id": "c:1"}).status_code
        == 404
    )
    # nothing was recorded by the failed attempts
    assert _next(client, uid).json()["pair_id"] == r["pair_id"]


def test_reveal_marks_which_yous_sided_with_the_pick(client, ctx):
    uid = _new_user(client, **MEN_APPAREL)
    client.put(f"/api/users/{uid}/style", json={"words": ["minimal"]})
    _play(client, uid, "train", domain="art")
    for chosen in ("a", "b"):
        r = _next(client, uid).json()
        preds = get_predictions(ctx.conn, r["pair_id"])
        assert "revealed" in preds
        resp = client.post(
            f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r[chosen]["id"]}
        ).json()
        expected = {
            you: (p > 0.5) == (chosen == "a") for you, p in preds.items() if p != 0.5
        }
        assert resp["reveal"] == expected
        assert all(isinstance(v, bool) for v in resp["reveal"].values())
        assert resp["progress"]["total"] == 6


def test_reveal_omits_abstaining_yous(client):
    uid = _ready(client, **MEN_APPAREL)  # no description: "stated" abstains
    r = _next(client, uid).json()
    reveal = client.post(
        f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r["a"]["id"]}
    ).json()["reveal"]
    assert "stated" not in reveal
    assert "revealed" not in reveal  # first five picks: revealed abstains


def test_reveal_is_null_outside_the_showdown(client):
    uid = _new_user(client)
    r = _next(client, uid, "art").json()
    resp = client.post(
        f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r["a"]["id"]}
    )
    assert resp.json()["reveal"] is None


# --- phases and errors -----------------------------------------------------------


def test_bad_domain_phase_or_combination_is_422(client):
    uid = _new_user(client)
    assert _next(client, uid, "shoes").status_code == 422
    assert _next(client, uid, phase="warmup").status_code == 422
    assert _next(client, uid, "art", "showdown").status_code == 422
    assert client.post(f"/api/users/{uid}/next", json={}).status_code == 422


def test_retest_is_not_available_yet(client):
    uid = _new_user(client)
    assert _next(client, uid, phase="retest").status_code == 501


def test_catalog_exhausted_is_409(client, ctx):
    ctx.settings = replace(ctx.settings, clothes_pairs=40)
    uid = _ready(client, **MEN_APPAREL)
    for _ in range(40):
        r = _next(client, uid)
        if r.status_code == 409:
            assert "unseen" in r.json()["detail"]
            return
        pair = r.json()
        client.post(
            f"/api/pairs/{pair['pair_id']}/choose", json={"chosen_id": pair["a"]["id"]}
        )
    pytest.fail("the catalog never ran out")


def test_unknown_user_is_404(client):
    ghost = "f" * 32
    assert _next(client, ghost).status_code == 404
    assert client.get(f"/api/users/{ghost}/results").status_code == 404
    assert (
        client.put(f"/api/users/{ghost}/style", json={"words": ["minimal"]}).status_code
        == 404
    )
    assert (
        client.put(f"/api/users/{ghost}/filters", json=MEN_APPAREL).status_code == 404
    )


def test_style_words_are_saved(client, ctx):
    uid = _new_user(client)
    words = ["minimal", "earthy"]
    assert client.put(f"/api/users/{uid}/style", json={"words": words}).json() == {}
    from amber.store.db import get_answers

    assert get_answers(ctx.conn, uid, "stated") == {"words": json.dumps(words)}


def test_style_rejects_zero_six_and_unknown_words(client, ctx):
    from amber.store.db import get_answers
    from amber.yous.stated import load_style_words

    uid = _new_user(client)
    six = load_style_words()[:6]
    for bad in ([], six, ["minimal", "not-a-style-word"]):
        r = client.put(f"/api/users/{uid}/style", json={"words": bad})
        assert r.status_code == 422
    assert get_answers(ctx.conn, uid, "stated") == {}
    five = load_style_words()[:5]
    assert client.put(f"/api/users/{uid}/style", json={"words": five}).status_code == 200


def test_get_style_words_returns_five_rows_of_six(client):
    from amber.yous.stated import load_style_words

    rows = client.get("/api/style-words").json()["rows"]
    assert [len(r) for r in rows] == [6] * 5
    assert [w for r in rows for w in r] == load_style_words()


def test_describe_route_is_gone(client):
    uid = _new_user(client)
    assert (
        client.put(f"/api/users/{uid}/describe", json={"text": "x"}).status_code in (404, 405)
    )


def test_results_before_any_clothes_pick(client):
    uid = _new_user(client)
    assert client.get(f"/api/users/{uid}/results").json() == {
        "scores_all": [],
        "scores_late": [],
        "curves": {},
        "n_clothes": 0,
    }


# --- plumbing --------------------------------------------------------------------


def test_requests_in_parallel_share_the_connection_safely(client):
    def one(i):
        uid = client.post("/api/users", json={"name": f"u{i}"}).json()["user_id"]
        client.put(f"/api/users/{uid}/filters", json=MEN_APPAREL)
        r = _next(client, uid, "art", "train")
        return r.status_code, uid

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, range(16)))
    assert {code for code, _ in results} == {200}
    assert len({uid for _, uid in results}) == 16


def test_cors_allows_the_vite_dev_server(client):
    r = client.options(
        "/api/users",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_serves_the_built_web_app_when_it_exists(ctx, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>amber</h1>")
    monkeypatch.setattr(app_module, "WEB_DIST", dist)
    client = TestClient(create_app(ctx))
    assert "<h1>amber</h1>" in client.get("/").text
    assert client.post("/api/users", json={"name": "x"}).status_code == 200
    assert client.get("/thumbs/clothes/1.jpg").status_code == 200


def test_no_web_dist_means_no_root_page(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path / "missing")
    assert TestClient(create_app(ctx)).get("/").status_code == 404


# --- one scored clothing phase ----------------------------------------------------


def test_clothes_showdown_waits_for_the_art_phase(client):
    uid = _new_user(client, **MEN_APPAREL)
    r = _next(client, uid)
    assert r.status_code == 409 and "art" in r.json()["detail"]
    _play(client, uid, "train", domain="art")
    assert _next(client, uid).status_code == 200


def test_clothes_train_is_a_422(client):
    uid = _ready(client, **MEN_APPAREL)
    assert _next(client, uid, phase="train").status_code == 422


def test_results_route_shape(client, ctx):
    uid = _ready(client, **MEN_APPAREL)
    n = ctx.settings.clothes_pairs
    assert len(_play(client, uid, "showdown")) == n
    res = client.get(f"/api/users/{uid}/results").json()
    assert set(res) == {"scores_all", "scores_late", "curves", "n_clothes"}
    assert res["n_clothes"] == n
    assert {s["you"] for s in res["scores_all"]} >= {"stated", "art", "revealed"}
    for s in res["scores_all"]:
        assert set(s) == {"you", "correct", "n", "accuracy", "ci_low", "ci_high"}
        assert s["n"] == n
    assert all(s["n"] == n - n // 2 for s in res["scores_late"])
    assert set(res["curves"]) == {s["you"] for s in res["scores_all"]}
    assert all(len(c) == n for c in res["curves"].values())


# --- the Big Five quiz -------------------------------------------------------------


def test_get_ipip_returns_twenty_items_in_file_order(client):
    from amber.yous.bigfive import load_ipip

    items = client.get("/api/ipip").json()["items"]
    assert items == [{"id": i.id, "text": i.text} for i in load_ipip()]


def test_ipip_answers_are_saved(client, ctx):
    from amber.store.db import get_answers

    uid = _new_user(client)
    answers = {"e1": 5, "a1": 1, "n2": 3}
    r = client.put(f"/api/users/{uid}/ipip", json={"answers": answers})
    assert r.status_code == 200 and r.json() == {"answered": 3}
    assert get_answers(ctx.conn, uid, "ipip") == {k: str(v) for k, v in answers.items()}


def test_ipip_accepts_an_empty_skip(client, ctx):
    uid = _new_user(client)
    r = client.put(f"/api/users/{uid}/ipip", json={"answers": {}})
    assert r.json() == {"answered": 0}


def test_ipip_rejects_unknown_items_and_bad_values(client, ctx):
    from amber.store.db import get_answers

    uid = _new_user(client)
    for bad in ({"zz": 3}, {"e1": 0}, {"e1": 6}, {"e1": 2, "x": 1}):
        r = client.put(f"/api/users/{uid}/ipip", json={"answers": bad})
        assert r.status_code == 422
    assert get_answers(ctx.conn, uid, "ipip") == {}
    ghost = "f" * 32
    assert (
        client.put(f"/api/users/{ghost}/ipip", json={"answers": {}}).status_code == 404
    )


def test_full_flow_with_quiz_scores_bigfive(client, ctx):
    from amber.yous.bigfive import load_ipip

    uid = _new_user(client, **MEN_APPAREL)
    answers = {i.id: (5 if i.keyed == 1 else 1) for i in load_ipip()}
    client.put(f"/api/users/{uid}/ipip", json={"answers": answers})
    _play(client, uid, "train", domain="art")
    _play(client, uid, "showdown")
    from amber.store.db import list_choices

    choices = list_choices(ctx.conn, uid, "clothes", "showdown")
    assert choices
    for c in choices:
        assert "bigfive" in get_predictions(ctx.conn, c.pair_id)
