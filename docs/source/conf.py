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
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/paramthakkar123/torchwm",
    "show_nav_level": 2,
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
    "navbar_center": ["navbar-nav"],
    "navigation_depth": 2,
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
