import tomllib
from pathlib import Path


def _dependency_names(dependencies):
    names = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.append(dependency.split(">=", maxsplit=1)[0])
        else:
            names.append(dependency["name"])
    return names


def test_jax_is_brax_optional_dependency_not_core_dependency():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "jax" not in _dependency_names(project["dependencies"])
    assert "jax" in _dependency_names(project["optional-dependencies"]["brax"])


def test_lockfile_keeps_jax_out_of_core_torchwm_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    torchwm = next(
        package for package in lock["package"] if package["name"] == "torchwm"
    )

    assert "jax" not in _dependency_names(torchwm["dependencies"])
    assert "jax" in _dependency_names(torchwm["optional-dependencies"]["brax"])
