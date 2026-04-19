"""Sphinx configuration for TorchWM documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "TorchWM"
copyright = f"{datetime.now().year}, Param Thakkar"
author = "Param Thakkar"
release = "0.2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid",
    "sphinx_thebe",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "deflist",
    "fieldlist",
]

autodoc_mock_imports = [
    "ale_py",
    "dm_control",
    "mlagents_envs",
    "moviepy",
    "cv2",
    "pygame",
    "hydra",
    "omegaconf",
    "torchvision",
    "gym",
    "gymnasium",
    "wandb",
    "PIL",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/paramthakkar123/torchwm",
    "show_nav_level": 1,
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
    "navbar_center": ["navbar-nav"],
    "navigation_depth": 2,
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo"],
    "navbar_align": "content",
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "switcher": {
        "json_url": "https://paramthakkar123.github.io/torchwm/switcher.json",
        "version_match": "stable",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchwm/",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/paramthakkar123/torchwm",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Thebe configuration for interactive code execution
thebe_config = {
    "repository_url": "https://github.com/paramthakkar123/torchwm",
    "repository_branch": "main",
    "selector": ".thebe, .cell",
    "always_load": False,
    "pre_execute": "",
    "post_execute": "",
    "request_kernel": True,
    "binder_options": {
        "repo": "paramthakkar123/torchwm",
        "ref": "main",
        "binder_url": "https://mybinder.org",
    },
}
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

# Canonical docs URL. Overridden in CI for PR previews.
html_baseurl = os.getenv(
    "SPHINX_HTML_BASEURL",
    "https://paramthakkar123.github.io/torchwm/",
)

# Open Graph metadata for social sharing cards/link previews.
ogp_site_url = html_baseurl
ogp_site_name = "TorchWM Documentation"
ogp_description_length = 200
ogp_enable_meta_description = True
