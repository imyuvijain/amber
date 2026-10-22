from amber.yous.registry import (
    BUILDERS,
    FROZEN_YOUS,
    build_all_yous,
    build_frozen_yous,
)


def test_stated_is_registered():
    assert "stated" in BUILDERS


def test_build_all_yous_runs_stated(ctx, user):
    yous = build_all_yous(ctx.conn, user, ctx)
    assert yous["stated"].name == "stated" and yous["stated"].w is None


def test_build_all_yous_isolates_failures(ctx, user, monkeypatch):
    monkeypatch.setitem(BUILDERS, "bigfive", lambda *_: 1 / 0)
    yous = build_all_yous(ctx.conn, user, ctx)
    assert yous["bigfive"].w is None and "error" in yous["bigfive"].meta
    assert yous["bigfive"].meta["error"].startswith("ZeroDivisionError: ")
    assert "stated" in yous  # the others still ran


def test_frozen_yous_are_stated_bigfive_art():
    assert FROZEN_YOUS == ("stated", "bigfive", "art")


def test_build_frozen_yous_builds_only_registered_frozen_ones(ctx, user):
    yous = build_frozen_yous(ctx.conn, user, ctx)
    assert set(yous) == set(FROZEN_YOUS) & set(BUILDERS)
    assert "revealed" not in yous


def test_build_frozen_yous_uses_a_bigfive_builder_once_registered(ctx, user, monkeypatch):
    from amber.yous.base import You

    monkeypatch.setitem(BUILDERS, "bigfive", lambda *_: You("bigfive", None, {}))
    assert "bigfive" in build_frozen_yous(ctx.conn, user, ctx)


def test_build_all_yous_includes_revealed(ctx, user):
    assert "revealed" in build_all_yous(ctx.conn, user, ctx)
