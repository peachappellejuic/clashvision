import pytest

from clashcv.config import ScreenConfig
from clashcv.demo import ELIXIR_BGR, H, W, card_image, fake_frame, sprite  # noqa: F401


@pytest.fixture
def cfg():
    return ScreenConfig()
