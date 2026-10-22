from fakes import red_and_blue_centered

from amber.store.db import save_answer
from amber.yous.base import predict
from amber.yous.stated import build_stated


def test_stated_abstains_without_description(ctx, user):
    you = build_stated(ctx.conn, user, ctx)
    assert you.name == "stated" and you.w is None
    assert you.meta == {"reason": "no description"}


def test_stated_abstains_on_blank_description(ctx, user):
    save_answer(ctx.conn, user, "describe", "text", "  \n ")
    assert build_stated(ctx.conn, user, ctx).meta == {"reason": "no description"}


def test_stated_prefers_red_items_when_asked_for_red(ctx, user):
    save_answer(ctx.conn, user, "describe", "text", "I love red clothes.")
    you = build_stated(ctx.conn, user, ctx)
    a, b = red_and_blue_centered(ctx)
    assert predict(you, a, b) == 1.0
    assert predict(you, b, a) == 0.0


def test_stated_splits_sentences(ctx, user):
    save_answer(
        ctx.conn, user, "describe", "text", "I like red! Blue is ok.\nNo green?"
    )
    you = build_stated(ctx.conn, user, ctx)
    assert you.meta["sentences"] == ["I like red", "Blue is ok", "No green"]
    assert you.w is not None


def test_stated_abstains_when_text_has_no_direction(ctx, user):
    # only punctuation: no sentences survive the split
    save_answer(ctx.conn, user, "describe", "text", "...!?")
    assert build_stated(ctx.conn, user, ctx).w is None
