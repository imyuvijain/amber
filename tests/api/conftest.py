from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from amber.api.app import create_app


@pytest.fixture
def client(ctx):
    ctx.settings = replace(ctx.settings, clothes_pairs=6)  # the fixture catalog is small
    return TestClient(create_app(ctx))
