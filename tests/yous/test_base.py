import numpy as np
from fakes import FakeEncoder

from amber.config import NEUTRAL_PROMPT
from amber.yous.base import YOU_ORDER, You, predict, text_direction


def test_you_order_lists_all_five_yous():
    assert YOU_ORDER == ("stated", "bigfive", "art", "values", "revealed")


def test_predict_sign_and_abstain():
    you = You("stated", np.array([1.0, 0.0]), {})
    assert predict(you, np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert predict(you, np.array([0.0, 1.0]), np.array([1.0, 0.0])) == 0.0
    assert predict(You("stated", None, {}), np.ones(2), np.zeros(2)) == 0.5


def test_predict_tie_is_half():
    you = You("stated", np.array([1.0, 0.0]), {})
    assert predict(you, np.array([0.5, 1.0]), np.array([0.5, -1.0])) == 0.5


def test_text_direction_is_unit_and_differs_from_neutral(fake_encoder):
    w = text_direction(fake_encoder, ["red things"])
    assert np.isclose(np.linalg.norm(w), 1)
    neutral = fake_encoder.embed_texts([NEUTRAL_PROMPT])[0]
    assert not np.allclose(w, neutral)


def test_text_direction_points_towards_the_named_colour(fake_encoder):
    w = text_direction(fake_encoder, ["red", "bright red"])
    assert w.argmax() == 0 and w[0] > 0.5


def test_text_direction_of_the_neutral_prompt_raises(fake_encoder):
    import pytest

    with pytest.raises(ValueError):
        text_direction(fake_encoder, [NEUTRAL_PROMPT])


def test_text_direction_rejects_empty_texts(fake_encoder):
    import pytest

    with pytest.raises(ValueError):
        text_direction(fake_encoder, [])


def test_fake_encoder_fixture_type(fake_encoder):
    assert isinstance(fake_encoder, FakeEncoder)
