import pytest
from fastapi.testclient import TestClient

from amber.api.app import create_app


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx))
