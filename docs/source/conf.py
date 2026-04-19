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
    # Enable the MathJax extension so Sphinx emits math nodes correctly. We also
    # include a small MathJax config file + runtime via `html_js_files` so the
    # client can render math reliably.
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
    "torch",
    "torch.nn",
    "torch._C",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/paramthakkar123/torchwm",
    "show_nav_level": 1,
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
    "navbar_center": [],
    # Keep only the theme icon links in the navbar end; the theme will render a
    # single primary search field by default. Removing the explicit search button
    # here avoids duplicate search controls in the header.
    "navbar_end": ["navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/paramthakkar123/torchwm",
            "icon": "fa-brands fa-github",
        },
    ],
}

# sphinxcontrib-mermaid: prefer raw output so the client-side mermaid.js can render
mermaid_output_format = "raw"

# Include client-side assets in a controlled order:
# 1) MathJax config + runtime so math renders reliably
# 2) Mermaid runtime + init so diagrams convert and render client-side
# 3) Thebe for runnable code blocks
html_js_files = [
    # MathJax config (local) and runtime (CDN)
    "mathjax_config.js",
    "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js",
    # Load Mermaid from jsDelivr CDN
    "https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.min.js",
    # Local init file that converts script blocks to .mermaid divs then runs Mermaid
    "mermaid_init.js",
    # Thebe (Thebe client) for runnable code blocks via Binder
    "https://unpkg.com/thebe@latest/lib/index.js",
    # Local small initializer to configure Thebe using `thebe_config`
    "thebe_init.js",
    "thebe_autoclass.js",
]

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
# Copy both our _static assets and the `images/` dir so images referenced
# from pages are available at build time (placed under _static/).
html_static_path = ["_static", "images"]

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
