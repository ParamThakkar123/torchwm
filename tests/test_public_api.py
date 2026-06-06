import torchwm
from torchwm import api


def test_top_level_torchwm_exports_user_facing_factories():
    assert torchwm.__version__ == "0.4.0"
    assert torchwm.create_config is api.create_config
    assert "dreamer" in torchwm.list_models()
    assert "gym" in torchwm.list_env_backends()


def test_create_config_accepts_aliases_and_overrides():
    cfg = torchwm.create_config("dreamerv1", env="cartpole-swingup", seed=123)
    assert cfg.env == "cartpole-swingup"
    assert cfg.seed == 123


def test_model_and_backend_specs_resolve_aliases():
    assert torchwm.get_model_spec("i-jepa").name == "jepa"
    assert torchwm.get_env_backend_spec("gymnasium").name == "gym"


def test_make_env_dispatches_to_selected_backend(monkeypatch):
    calls = {}

    def fake_loader(import_path):
        calls["import_path"] = import_path

        def factory(env_id, **kwargs):
            return {"env_id": env_id, "kwargs": kwargs}

        return factory

    monkeypatch.setattr(api, "_load_object", fake_loader)
    env = api.make_env("CartPole-v1", backend="gym", render_mode="rgb_array")

    assert calls["import_path"] == "world_models.envs:make_gym_env"
    assert env == {
        "env_id": "CartPole-v1",
        "kwargs": {"render_mode": "rgb_array"},
    }


def test_create_model_for_factory_only_spec_filters_through_signature(monkeypatch):
    spec = api.ModelSpec(
        name="dummy",
        import_path="tests.dummy:create_dummy",
        description="Test-only factory",
    )

    def fake_loader(import_path):
        assert import_path == spec.import_path

        def create_dummy(required, optional=1):
            return {"required": required, "optional": optional}

        return create_dummy

    monkeypatch.setitem(api.MODEL_SPECS, "dummy", spec)
    monkeypatch.setattr(api, "_load_object", fake_loader)

    assert api.create_model("dummy", required=3, optional=5) == {
        "required": 3,
        "optional": 5,
    }


def test_layer_and_helper_packages_are_importable():
    import world_models.helpers as helpers
    from world_models.layers import AdaLNNormalization, RMSNorm

    assert "load_checkpoint" in dir(helpers)
    assert RMSNorm.__name__ == "RMSNorm"
    assert AdaLNNormalization.__name__ == "AdaLNNormalization"
