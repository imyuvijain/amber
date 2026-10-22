from amber.yous.registry import BUILDERS, build_all_yous


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
