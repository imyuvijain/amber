import numpy as np
import pytest
from fakes import play_art, play_showdown, play_train, red_and_blue_centered

from amber.yous.art_taste import art_direction, build_art
from amber.yous.base import predict
from amber.yous.registry import BUILDERS


def test_art_is_registered():
    assert "art" in BUILDERS


def test_art_direction_rejects_clothes_choices(ctx, user):
    clothes = play_train(ctx, user, prefer="red", n=1)
    with pytest.raises(ValueError):
        art_direction(clothes, ctx.index["art"], ctx.basis["art"])


def test_art_direction_rejects_a_mixed_list(ctx, user):
    mixed = play_art(ctx, user, prefer="red", n=5) + play_train(ctx, user, n=1)
    with pytest.raises(ValueError):
        art_direction(mixed, ctx.index["art"], ctx.basis["art"])


def test_art_direction_none_when_short(ctx, user):
    few = play_art(ctx, user, prefer="red", n=4)
    assert art_direction(few, ctx.index["art"], ctx.basis["art"]) is None


def test_art_direction_is_unit_in_full_space(ctx, user):
    chosen = play_art(ctx, user, prefer="red", n=6)
    w = art_direction(chosen, ctx.index["art"], ctx.basis["art"])
    assert w.shape == (ctx.index["art"].vectors.shape[1],)
    assert np.isclose(np.linalg.norm(w), 1.0, atol=1e-5)


def test_art_you_is_unit_and_abstains_when_short(ctx, user):
    play_art(ctx, user, prefer="red", n=4)
    short = build_art(ctx.conn, user, ctx)
    assert short.name == "art" and short.w is None
    assert short.meta["reason"] == "fewer than 5 art choices"
    play_art(ctx, user, prefer="red", n=2)  # now 6 total
    you = build_art(ctx.conn, user, ctx)
    assert np.isclose(np.linalg.norm(you.w), 1)
    assert you.meta == {"n": 6}


def test_art_you_abstains_with_no_choices(ctx, user):
    assert build_art(ctx.conn, user, ctx).w is None


def test_red_art_lover_predicts_red_clothes(ctx, user):
    # FakeEncoder shares the RGB dims across domains
    play_art(ctx, user, prefer="red", n=6)
    assert predict(build_art(ctx.conn, user, ctx), *red_and_blue_centered(ctx)) == 1.0


def test_blue_art_lover_predicts_blue_clothes(ctx, user):
    play_art(ctx, user, prefer="blue", n=6)
    assert predict(build_art(ctx.conn, user, ctx), *red_and_blue_centered(ctx)) == 0.0


def test_art_you_ignores_clothes_choices(ctx, user):
    play_train(ctx, user, prefer="blue", n=6)
    play_showdown(ctx, user, prefer="blue", n=3)
    assert build_art(ctx.conn, user, ctx).w is None
    play_art(ctx, user, prefer="red", n=6)
    you = build_art(ctx.conn, user, ctx)
    assert you.meta["n"] == 6
    assert predict(you, *red_and_blue_centered(ctx)) == 1.0
