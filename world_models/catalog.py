"""Auto-discovery system for models and environments.

This module automatically discovers available models, environments, and their
compatibility by scanning the codebase. New models and environments are
automatically added when they are created in the appropriate directories.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from gymnasium.envs.registration import registry as gym_registry


def discover_modules(package_path: str) -> dict[str, Any]:
    """Discover all modules in a package.

    Args:
        package_path: Dot-separated package path (e.g., 'world_models.models')

    Returns:
        Dict mapping module names to imported modules
    """
    try:
        package = importlib.import_module(package_path)
    except ImportError:
        return {}

    modules = {}
    package_dir = Path(package.__file__).parent

    for _, name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if name.startswith("_"):
            continue
        full_name = f"{package_path}.{name}"
        try:
            modules[name] = importlib.import_module(full_name)
        except ImportError:
            pass

    return modules


def get_model_metadata(model_class: type) -> dict[str, Any]:
    """Extract metadata from a model class.

    Looks for:
    - _supported_environments: List of environment names
    - _default_config: Default configuration dict
    - _description: Model description
    - train function in training module

    Args:
        model_class: Model class to inspect

    Returns:
        Dict with model metadata
    """
    metadata = {
        "supported_environments": getattr(model_class, "_supported_environments", None),
        "default_config": getattr(model_class, "_default_config", {}),
        "description": getattr(model_class, "__doc__", ""),
        "has_training": False,
    }

    if metadata["description"]:
        metadata["description"] = metadata["description"].strip().split("\n")[0]

    return metadata


def discover_environments() -> dict[str, list[str]]:
    """Discover all available environments.

    Returns:
        Dict with environment categories and their environments
    """
    environments = {
        "gymnasium": [],
        "dm_control": [],
        "atari": [],
        "custom": [],
    }

    for env_id in gym_registry:
        env_spec = gym_registry[env_id]
        if env_spec is None:
            continue

        name = env_spec.id

        if "dm_control" in name or "dmc" in name:
            environments["dm_control"].append(name)
        elif "ALE" in name or "Atari" in name:
            environments["atari"].append(name)
        elif "CarRacing" in name:
            environments["custom"].append(name)
        else:
            environments["gymnasium"].append(name)

    for key in environments:
        environments[key] = sorted(set(environments[key]))

    return environments


def discover_models() -> dict[str, dict[str, Any]]:
    """Discover all available models by scanning models directory.

    Returns:
        Dict mapping model names to their metadata
    """
    models = {}

    model_files = [
        ("dreamer", "DreamerAgent", "world_models.models.dreamer"),
        ("planet", "Planet", "world_models.models.planet"),
        ("jepa", "JEPAAgent", "world_models.models.jepa_agent"),
        ("controller", "Controller", "world_models.models.controller"),
        ("mdrnn", "MDRNN", "world_models.models.mdrnn"),
    ]

    for model_key, class_name, module_path in model_files:
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name, None)

            if model_class is None:
                continue

            metadata = get_model_metadata(model_class)
            models[model_key] = {
                "label": class_name,
                "class_name": class_name,
                "module": module_path,
                "description": metadata.get("description", ""),
                "default_config": metadata.get("default_config", {}),
                "supported_environments": metadata.get("supported_environments", None),
                "has_training": metadata.get("has_training", False),
            }
        except ImportError:
            pass

    return models


def infer_model_environments(model_key: str, model_class: type) -> list[str]:
    """Infer supported environments for a model based on training code.

    Args:
        model_key: Model identifier
        model_class: Model class

    Returns:
        List of supported environment names
    """
    envs = []

    train_module_map = {
        "dreamer": "world_models.training.train_dreamer",
        "planet": "world_models.training.train_planet",
        "jepa": "world_models.training.train_jepa",
    }

    if model_key in train_module_map:
        try:
            train_module = importlib.import_module(train_module_map[model_key])

            for attr_name in dir(train_module):
                attr = getattr(train_module, attr_name)
                if callable(attr) and hasattr(attr, "__code__"):
                    source = inspect.getsource(attr)

                    if "dm_control" in source or "dmc" in source:
                        envs.extend(
                            [
                                "cartpole-balance",
                                "cartpole-swingup",
                                "cheetah-run",
                                "finger-spin",
                                "reacher-easy",
                                "walker-walk",
                                "walker-run",
                                "quadruped-walk",
                            ]
                        )

                    if "gym.make" in source or "make_vec" in source:
                        import re

                        matches = re.findall(r'["\']([\w\-]+)-v\d+["\']', source)
                        envs.extend(matches)

        except ImportError:
            pass

    return list(set(envs))


def build_catalog() -> dict[str, Any]:
    """Build the complete catalog of models and environments.

    This function scans the codebase and builds a complete catalog
    that can be used by the UI.

    Returns:
        Dict containing models, environments, and their relationships
    """
    models = discover_models()
    environments = discover_environments()

    for model_key, model_info in models.items():
        if model_info.get("supported_environments") is None:
            try:
                module = importlib.import_module(model_info["module"])
                model_class = getattr(module, model_info["class_name"])
                inferred = infer_model_environments(model_key, model_class)
                if inferred:
                    model_info["supported_environments"] = inferred
                else:
                    model_info["supported_environments"] = environments["gymnasium"][
                        :50
                    ]
            except ImportError:
                model_info["supported_environments"] = []

    default_training_configs = {
        "dreamer": {
            "total_steps": 20000,
            "seed_steps": 1000,
            "update_steps": 50,
            "collect_steps": 500,
            "test_interval": 2000,
            "test_episodes": 1,
        },
        "planet": {
            "epochs": 25,
            "warmup_episodes": 2,
            "steps_per_epoch": 50,
            "batch_size": 32,
            "horizon": 50,
            "beta": 1.0,
        },
        "jepa": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-4,
        },
    }

    return {
        "models": models,
        "environments": environments,
        "default_training_configs": default_training_configs,
    }


CATALOG = build_catalog()


def get_catalog() -> dict[str, Any]:
    """Get the current catalog (cached).

    Returns:
        The catalog dict
    """
    return CATALOG


def refresh_catalog() -> dict[str, Any]:
    """Force refresh the catalog.

    Returns:
        The refreshed catalog dict
    """
    global CATALOG
    CATALOG = build_catalog()
    return CATALOG
