import numpy as np
from amber.yous.revealed import build_revealed, train_preference_model
from fakes import play_showdown, play_train, red_and_blue_centered

from amber.config import PRIOR_VAR
from amber.store.db import answer_pair, create_pair, list_choices
from amber.yous.base import predict
from amber.yous.registry import BUILDERS


def test_revealed_is_registered():
    assert "revealed" in BUILDERS


def test_build_revealed_ignores_showdown_choices(ctx, user):
    play_train(ctx, user, prefer="red", n=6)
    play_showdown(ctx, user, prefer="blue", n=3)
    you = build_revealed(ctx.conn, user, ctx)
    assert you.meta["n"] == 6


def test_revealed_abstains_under_5(ctx, user):
    play_train(ctx, user, n=4)
    you = build_revealed(ctx.conn, user, ctx)
    assert you.name == "revealed" and you.w is None and you.meta["n"] == 4


def test_revealed_abstains_with_no_choices(ctx, user):
    you = build_revealed(ctx.conn, user, ctx)
    assert you.w is None and you.meta["n"] == 0


def test_revealed_speaks_from_5_choices(ctx, user):
    play_train(ctx, user, prefer="red", n=5)
    assert build_revealed(ctx.conn, user, ctx).w is not None


def test_revealed_learns_simulated_preference(ctx, user):
    play_train(ctx, user, prefer="red", n=6)  # FakeEncoder: dim 0 is red
    you = build_revealed(ctx.conn, user, ctx)
    assert predict(you, *red_and_blue_centered(ctx)) == 1.0


def test_revealed_learns_the_opposite_preference_too(ctx, user):
    play_train(ctx, user, prefer="blue", n=6)
    you = build_revealed(ctx.conn, user, ctx)
    assert predict(you, *red_and_blue_centered(ctx)) == 0.0


def test_revealed_w_is_a_unit_vector_in_full_space(ctx, user):
    play_train(ctx, user, prefer="red", n=6)
    w = build_revealed(ctx.conn, user, ctx).w
    assert w.shape == (ctx.index["clothes"].vectors.shape[1],)
    assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-5)


def test_revealed_ignores_art_choices(ctx, user):
    from fakes import play_art

    play_art(ctx, user, prefer="red", n=6)
    assert build_revealed(ctx.conn, user, ctx).meta["n"] == 0


def test_train_preference_model_replays_answered_choices_in_order(ctx, user):
    choices = play_train(ctx, user, prefer="red", n=4)
    model = train_preference_model(choices, ctx.index["clothes"], ctx.basis["clothes"])
    assert model.n_updates == 4
    expected = train_preference_model(
        list(choices), ctx.index["clothes"], ctx.basis["clothes"], PRIOR_VAR
    )
    assert np.allclose(model.mean, expected.mean)


def test_train_preference_model_skips_unanswered_choices(ctx, user):
    answered = play_train(ctx, user, n=2)
    open_ = create_pair(ctx.conn, user, "clothes", "train", "c:90", "c:91")
    model = train_preference_model(
        answered + [open_], ctx.index["clothes"], ctx.basis["clothes"]
    )
    assert model.n_updates == 2


def test_train_preference_model_pushes_winner_up(ctx, user):
    pair = create_pair(ctx.conn, user, "clothes", "train", "c:1", "c:2")
    done = answer_pair(ctx.conn, pair.pair_id, "c:1")
    model = train_preference_model([done], ctx.index["clothes"], ctx.basis["clothes"])
    win, lose = ctx.basis["clothes"].project(
        ctx.index["clothes"].centered(["c:1", "c:2"])
    )
    assert float(model.mean @ (win - lose)) > 0
    assert len(list_choices(ctx.conn, user)) == 1
