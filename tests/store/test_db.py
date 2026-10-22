from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from amber.store import db
from amber.store.db import (
    AlreadyAnswered,
    Choice,
    MissingPredictions,
    NotFound,
    StoreError,
    answer_pair,
    connect,
    create_pair,
    create_user,
    get_answers,
    get_predictions,
    list_choices,
    load_yous,
    open_pair,
    save_answer,
    save_predictions,
    save_you,
    shown_ids,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)


def test_exceptions_share_a_base():
    for exc in (AlreadyAnswered, MissingPredictions, NotFound):
        assert issubclass(exc, StoreError)


def test_connect_file_creates_schema_and_persists(tmp_path):
    path = tmp_path / "amber.db"
    c = connect(path)
    u = create_user(c, "y")
    save_answer(c, u, "describe", "q1", "hello")
    c.close()
    c2 = connect(path)
    assert get_answers(c2, u, "describe") == {"q1": "hello"}
    c2.close()


def test_create_user_returns_distinct_uuid_hex(conn):
    a, b = create_user(conn, "a"), create_user(conn, "b")
    assert a != b and len(a) == 32 and int(a, 16) >= 0


def test_save_answer_upserts_and_isolates_kinds_and_users(conn):
    u, v = create_user(conn, "u"), create_user(conn, "v")
    save_answer(conn, u, "ipip", "e1", "3")
    save_answer(conn, u, "ipip", "e1", "5")
    save_answer(conn, u, "ipip", "e2", "1")
    save_answer(conn, u, "values", "e1", "x")
    save_answer(conn, v, "ipip", "e1", "2")
    assert get_answers(conn, u, "ipip") == {"e1": "5", "e2": "1"}
    assert get_answers(conn, u, "values") == {"e1": "x"}
    assert get_answers(conn, v, "ipip") == {"e1": "2"}
    assert get_answers(conn, u, "filters") == {}


def test_answer_is_idempotent(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    first = answer_pair(conn, p.pair_id, "c:1")
    again = answer_pair(conn, p.pair_id, "c:1", now=T0 + timedelta(hours=1))
    assert first == again and len(list_choices(conn, u)) == 1
    assert again.answered_at == first.answered_at


def test_conflicting_answer_rejected(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    answer_pair(conn, p.pair_id, "c:1")
    with pytest.raises(AlreadyAnswered):
        answer_pair(conn, p.pair_id, "c:2")
    assert list_choices(conn, u)[0].chosen_id == "c:1"


def test_chosen_must_be_in_pair(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    with pytest.raises(ValueError):
        answer_pair(conn, p.pair_id, "c:9")
    assert list_choices(conn, u) == []


def test_unknown_pair_raises_not_found(conn):
    with pytest.raises(NotFound):
        answer_pair(conn, "nope", "c:1")
    with pytest.raises(NotFound):
        save_predictions(conn, "nope", {"stated": 0.5})


def test_open_pair_resumes(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "art", "train", "a:1", "a:2", now=T0)
    assert open_pair(conn, u, "art", "train") == p
    assert p.chosen_id is None and p.answered_at is None and p.shown_at == T0


def test_open_pair_none_when_answered_or_other_slot(conn):
    u, v = create_user(conn, "y"), create_user(conn, "z")
    assert open_pair(conn, u, "art", "train") is None
    p = create_pair(conn, u, "art", "train", "a:1", "a:2")
    assert open_pair(conn, u, "clothes", "train") is None
    assert open_pair(conn, u, "art", "showdown") is None
    assert open_pair(conn, v, "art", "train") is None
    answer_pair(conn, p.pair_id, "a:1")
    assert open_pair(conn, u, "art", "train") is None


def test_open_pair_returns_most_recent_unanswered(conn):
    u = create_user(conn, "y")
    create_pair(conn, u, "art", "train", "a:1", "a:2", now=T0)
    newer = create_pair(
        conn, u, "art", "train", "a:3", "a:4", now=T0 + timedelta(seconds=1)
    )
    assert open_pair(conn, u, "art", "train") == newer


def test_create_pair_retest_of_and_default_now_is_utc(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    r = create_pair(conn, u, "clothes", "retest", "c:2", "c:1", retest_of=p.pair_id)
    assert r.retest_of == p.pair_id and p.retest_of is None
    assert r.shown_at.utcoffset() == timedelta(0)
    assert open_pair(conn, u, "clothes", "retest") == r


def test_datetimes_round_trip_exactly(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2", now=T0)
    done = answer_pair(conn, p.pair_id, "c:2", now=T0 + timedelta(seconds=3))
    assert done.shown_at == T0 and done.answered_at == T0 + timedelta(seconds=3)
    assert list_choices(conn, u) == [done]
    assert isinstance(done, Choice)
    with pytest.raises(AttributeError):
        done.chosen_id = "c:1"  # frozen


def test_list_choices_oldest_first_and_filters(conn):
    u = create_user(conn, "y")
    # same shown_at: insertion order breaks the tie
    p1 = create_pair(conn, u, "clothes", "train", "c:1", "c:2", now=T0)
    p2 = create_pair(conn, u, "clothes", "train", "c:3", "c:4", now=T0)
    p0 = create_pair(conn, u, "art", "train", "a:1", "a:2", now=T0 - timedelta(days=1))
    p3 = create_pair(
        conn, u, "clothes", "showdown", "c:5", "c:6", now=T0 + timedelta(days=1)
    )
    save_predictions(conn, p3.pair_id, {"stated": 0.5})
    open_ = create_pair(
        conn, u, "clothes", "train", "c:7", "c:8", now=T0 + timedelta(days=2)
    )
    for p in (p1, p2, p0):
        answer_pair(conn, p.pair_id, p.a_id)
    answer_pair(conn, p3.pair_id, "c:6")

    assert [c.pair_id for c in list_choices(conn, u)] == [
        p0.pair_id,
        p1.pair_id,
        p2.pair_id,
        p3.pair_id,
    ]
    assert [c.pair_id for c in list_choices(conn, u, domain="clothes")] == [
        p1.pair_id,
        p2.pair_id,
        p3.pair_id,
    ]
    assert [c.pair_id for c in list_choices(conn, u, "clothes", "showdown")] == [
        p3.pair_id
    ]
    assert [c.pair_id for c in list_choices(conn, u, phase="train")] == [
        p0.pair_id,
        p1.pair_id,
        p2.pair_id,
    ]
    everything = list_choices(conn, u, answered_only=False)
    assert everything[-1].pair_id == open_.pair_id and len(everything) == 5
    assert all(c.chosen_id is not None for c in list_choices(conn, u))


def test_list_choices_scoped_to_user(conn):
    u, v = create_user(conn, "y"), create_user(conn, "z")
    p = create_pair(conn, u, "art", "train", "a:1", "a:2")
    answer_pair(conn, p.pair_id, "a:1")
    assert list_choices(conn, v) == []


def test_shown_ids_any_phase_answered_or_not_scoped(conn):
    u, v = create_user(conn, "y"), create_user(conn, "z")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    answer_pair(conn, p.pair_id, "c:1")
    create_pair(conn, u, "clothes", "retest", "c:3", "c:4")  # unanswered
    create_pair(conn, u, "art", "train", "a:1", "a:2")
    create_pair(conn, v, "clothes", "train", "c:9", "c:8")
    assert shown_ids(conn, u, "clothes") == {"c:1", "c:2", "c:3", "c:4"}
    assert shown_ids(conn, u, "art") == {"a:1", "a:2"}
    assert shown_ids(conn, create_user(conn, "w"), "art") == set()


def test_showdown_requires_predictions(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    with pytest.raises(MissingPredictions):
        answer_pair(conn, p.pair_id, "c:1")
    assert list_choices(conn, u) == []
    save_predictions(conn, p.pair_id, {"stated": 0.6})
    assert answer_pair(conn, p.pair_id, "c:1").chosen_id == "c:1"


def test_train_pair_needs_no_predictions(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    assert answer_pair(conn, p.pair_id, "c:2").chosen_id == "c:2"


def test_validation_precedes_prediction_gate(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    with pytest.raises(ValueError):
        answer_pair(conn, p.pair_id, "c:9")


def test_predictions_roundtrip_and_overwrite(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    assert get_predictions(conn, p.pair_id) == {}
    save_predictions(conn, p.pair_id, {"stated": 0.6, "revealed": 0.25})
    assert get_predictions(conn, p.pair_id) == {"stated": 0.6, "revealed": 0.25}
    save_predictions(conn, p.pair_id, {"stated": 0.7})
    assert get_predictions(conn, p.pair_id)["stated"] == 0.7
    assert get_predictions(conn, p.pair_id)["revealed"] == 0.25


def test_foreign_keys_enforced(conn):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        create_pair(conn, "ghost", "clothes", "train", "c:1", "c:2")


def test_yous_roundtrip(conn):
    u = create_user(conn, "y")
    save_you(conn, u, "revealed", np.ones(4, np.float32) / 2, {"n": 40})
    save_you(conn, u, "values", None, {"reason": "no key"})
    w, meta = load_yous(conn, u)["revealed"]
    assert np.allclose(w, 0.5) and meta == {"n": 40} and w.dtype == np.float32
    assert load_yous(conn, u)["values"][0] is None
    assert load_yous(conn, u)["values"][1] == {"reason": "no key"}


def test_save_you_upserts_and_casts_to_float32(conn):
    u, v = create_user(conn, "y"), create_user(conn, "z")
    save_you(conn, u, "revealed", np.zeros(3, np.float64), {"n": 1})
    save_you(conn, u, "revealed", np.arange(3, dtype=np.float64), {"n": 2})
    w, meta = load_yous(conn, u)["revealed"]
    assert w.dtype == np.float32 and w.tolist() == [0.0, 1.0, 2.0] and meta == {"n": 2}
    assert len(load_yous(conn, u)) == 1 and load_yous(conn, v) == {}


def test_predictions_rejected_after_answer_and_left_unchanged(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    save_predictions(conn, p.pair_id, {"stated": 0.6, "revealed": 0.25})
    answer_pair(conn, p.pair_id, "c:1")
    with pytest.raises(AlreadyAnswered):
        save_predictions(conn, p.pair_id, {"stated": 0.99, "art": 0.1})
    assert get_predictions(conn, p.pair_id) == {"stated": 0.6, "revealed": 0.25}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_predictions_must_be_finite_probabilities(conn, bad):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    with pytest.raises(ValueError):
        save_predictions(conn, p.pair_id, {"stated": 0.5, "revealed": bad})
    assert get_predictions(conn, p.pair_id) == {}


def test_predictions_boundaries_allowed(conn):
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    save_predictions(conn, p.pair_id, {"a": 0.0, "b": 1.0})
    assert get_predictions(conn, p.pair_id) == {"a": 0.0, "b": 1.0}


def test_answer_pair_lost_race_same_id_returns_pair_other_id_raises(conn, monkeypatch):
    """Simulate another writer answering between our read and our guarded UPDATE."""
    u = create_user(conn, "y")
    p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    real_get = db._get_pair
    calls = {"n": 0}

    def stale_get(c, pair_id):
        calls["n"] += 1
        if calls["n"] == 1:  # first read sees it unanswered, then the rival writes
            pair = real_get(c, pair_id)
            c.execute(
                "UPDATE pairs SET chosen_id = 'c:1', answered_at = ? WHERE pair_id = ?",
                (db._iso(T0), pair_id),
            )
            return pair
        return real_get(c, pair_id)

    monkeypatch.setattr(db, "_get_pair", stale_get)
    got = answer_pair(conn, p.pair_id, "c:1")
    assert got.chosen_id == "c:1" and got.answered_at == T0

    p2 = create_pair(conn, u, "clothes", "train", "c:3", "c:4")
    calls["n"] = 0

    def stale_get2(c, pair_id):
        calls["n"] += 1
        if calls["n"] == 1:
            pair = real_get(c, pair_id)
            c.execute(
                "UPDATE pairs SET chosen_id = 'c:3', answered_at = ? WHERE pair_id = ?",
                (db._iso(T0), pair_id),
            )
            return pair
        return real_get(c, pair_id)

    monkeypatch.setattr(db, "_get_pair", stale_get2)
    with pytest.raises(AlreadyAnswered):
        answer_pair(conn, p2.pair_id, "c:4")
