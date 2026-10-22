from amber.store.db import connect


def test_ctx_has_both_domains(ctx):
    for domain in ("clothes", "art"):
        assert len(ctx.items[domain]) == len(ctx.index[domain]) > 0
        assert ctx.basis[domain].components.shape[1] == ctx.encoder.dim
        assert ctx.basis2d[domain].components.shape == (2, ctx.encoder.dim)
    assert ctx.llm is None
    assert ctx.settings.showdown_pairs == 3


def test_load_context_opens_db_in_data_dir(ctx, fixture_data):
    from amber.context import load_context

    again = load_context(fixture_data, encoder=ctx.encoder)
    try:
        assert (fixture_data / "amber.db").exists()
        assert again.data_dir == fixture_data
    finally:
        again.conn.close()
    assert connect(":memory:") is not None
