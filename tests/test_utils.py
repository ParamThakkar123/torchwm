import sys
import types

import pytest

try:
    import attrdict  # noqa: F401
except ModuleNotFoundError:
    module = types.ModuleType("attrdict")

    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    module.AttrDict = AttrDict
    sys.modules["attrdict"] = module

from world_models.utils.utils import load_yml_config


def test_load_yml_config_returns_full_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("train:\n  lr: 0.001\nname: torchwm\n", encoding="utf-8")

    cfg = load_yml_config(cfg_path)

    assert cfg["name"] == "torchwm"
    assert cfg["train"]["lr"] == 0.001


def test_load_yml_config_returns_specific_variable(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("train:\n  lr: 0.001\nname: torchwm\n", encoding="utf-8")

    assert load_yml_config(cfg_path, variable="name") == "torchwm"
    assert load_yml_config(cfg_path, variable="train.lr") == 0.001

    train_cfg = load_yml_config(cfg_path, variable="train")
    assert train_cfg["lr"] == 0.001


def test_load_yml_config_variable_missing_raises_key_error(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("train:\n  lr: 0.001\n", encoding="utf-8")

    with pytest.raises(KeyError, match="not found"):
        load_yml_config(cfg_path, variable="train.missing")
