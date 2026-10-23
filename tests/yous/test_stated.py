import json

from fakes import red_and_blue_centered

from amber.store.db import save_answer
from amber.yous.base import predict
from amber.yous.stated import build_stated, load_style_words


def _words(ctx, user, words):
    save_answer(ctx.conn, user, "stated", "words", json.dumps(words))


def test_style_words_are_thirty_unique_lowercase_phrases():
    words = load_style_words()
    assert len(words) == 30 and len(set(words)) == 30
    assert all(w == w.lower() and 1 <= len(w.split()) <= 2 for w in words)


def test_stated_abstains_without_words(ctx, user):
    you = build_stated(ctx.conn, user, ctx)
    assert you.name == "stated" and you.w is None
    assert you.meta == {"reason": "no words"}


def test_stated_abstains_on_empty_word_list(ctx, user):
    _words(ctx, user, [])
    assert build_stated(ctx.conn, user, ctx).meta == {"reason": "no words"}


def test_stated_prefers_red_items_when_the_word_is_red(ctx, user):
    _words(ctx, user, ["red"])
    you = build_stated(ctx.conn, user, ctx)
    a, b = red_and_blue_centered(ctx)
    assert predict(you, a, b) == 1.0
    assert predict(you, b, a) == 0.0
    assert you.meta == {"words": ["red"]}


def test_stated_ignores_the_old_describe_text(ctx, user):
    save_answer(ctx.conn, user, "describe", "text", "I love red clothes.")
    assert build_stated(ctx.conn, user, ctx).meta == {"reason": "no words"}
