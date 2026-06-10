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


def test_export_model_torchscript_writes_file(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    import world_models.export  # noqa: F401 - installs torch.nn.Module.export

    class TinyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    agent = TinyAgent()
    path = agent.export(
        tmp_path / "tiny.pt",
        format="torchscript",
        example_inputs=torch.zeros(1, 2),
    )

    assert path.exists()
    loaded = torch.jit.load(str(path))
    assert loaded(torch.zeros(1, 2)).shape == (1, 1)


def test_top_level_exports_export_helpers():
    import pytest
    import torchwm

    pytest.importorskip("torch")
    from world_models.export import ExportableAgentMixin, export_any, export_model

    assert torchwm.export_any is export_any
    assert torchwm.export_model is export_model
    assert torchwm.ExportableAgentMixin is ExportableAgentMixin


def test_layer_and_helper_packages_are_importable():
    import world_models.helpers as helpers
    from world_models.layers import AdaLNNormalization, RMSNorm

    assert "load_checkpoint" in dir(helpers)
    assert RMSNorm.__name__ == "RMSNorm"
    assert AdaLNNormalization.__name__ == "AdaLNNormalization"


def test_diamond_and_dit_are_registered_in_public_api():
    assert "diamond" in torchwm.list_models()
    assert "dit" in torchwm.list_models()

    diamond_cfg = torchwm.create_config("diamond", game="Pong-v5", seed=11)
    dit_cfg = torchwm.create_config("diffusion-transformer", IMG_SIZE=8, PATCH=4)

    assert diamond_cfg.game == "Pong-v5"
    assert diamond_cfg.seed == 11
    assert dit_cfg.IMG_SIZE == 8
    assert dit_cfg.PATCH == 4
    assert torchwm.get_model_spec("diamond_agent").name == "diamond"
    assert torchwm.get_model_spec("diffusion_transformer").name == "dit"


def test_create_model_uses_dit_config_adapter():
    model = torchwm.create_model(
        "dit",
        IMG_SIZE=8,
        PATCH=4,
        CHANNELS=3,
        WIDTH=16,
        DEPTH=1,
        HEADS=4,
        DROP=0.0,
    )

    assert model.patchify.proj.in_channels == 3
    assert len(model.transformer_blocks) == 1


def test_create_model_dispatches_diamond_agent_with_config(monkeypatch):
    captured = {}

    class FakeDiamondAgent:
        def __init__(self, config):
            captured["config"] = config

    original_loader = api._load_object

    def fake_loader(import_path):
        if import_path == "world_models.training.train_diamond:DiamondAgent":
            return FakeDiamondAgent
        return original_loader(import_path)

    monkeypatch.setattr(api, "_load_object", fake_loader)
    agent = api.create_model("diamond", game="Pong-v5")

    assert isinstance(agent, FakeDiamondAgent)
    assert captured["config"].game == "Pong-v5"
