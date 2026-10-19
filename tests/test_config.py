from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from craton.config import (
    CratonConfig,
    ModelShape,
    available_configs,
    config_hash,
    load_config,
    trunk_params,
)


def test_every_shipped_config_loads() -> None:
    assert available_configs() == ["mini", "planet", "regional"]
    for name in available_configs():
        load_config(name)


def test_mini_fits_the_laptop_story() -> None:
    mini = load_config("mini")
    assert mini.status == "active"
    assert 2_000_000 <= mini.params <= 5_000_000
    assert mini.train.precision == "fp16"
    assert len(mini.train.seeds) >= 3


def test_twelve_d_squared_rule() -> None:
    shape = ModelShape(d_model=768, n_layers=12, n_heads=12, context_tokens=1024)
    n = trunk_params(shape)
    assert abs(n - 12 * 768**2 * 12) / n < 0.01  # GPT-2 small's trunk, ~85M


def _mini_dict() -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs" / "mini.yaml"
    data: dict[str, object] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def test_unknown_keys_rejected() -> None:
    data = _mini_dict()
    data["learning_rate_typo"] = 1
    with pytest.raises(ValidationError):
        CratonConfig.model_validate(data)


def test_dishonest_size_rejected() -> None:
    data = _mini_dict()
    data["params_range"] = [10, 20]
    with pytest.raises(ValidationError, match="claims"):
        CratonConfig.model_validate(data)


def test_heads_must_divide_width() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelShape(d_model=256, n_layers=2, n_heads=7, context_tokens=8)


def test_too_few_seeds_rejected() -> None:
    data = _mini_dict()
    train = dict(data["train"])  # type: ignore[call-overload]
    train["seeds"] = [0, 1]
    data["train"] = train
    with pytest.raises(ValidationError):
        CratonConfig.model_validate(data)


def test_config_hash_tracks_content() -> None:
    mini = load_config("mini")
    assert config_hash(mini) == config_hash(load_config("mini"))
    assert config_hash(mini) != config_hash(load_config("regional"))
