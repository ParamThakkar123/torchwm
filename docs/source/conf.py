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
    # We load MathJax manually (config + runtime) so we control ordering and
    # avoid Sphinx injecting the runtime before our config. Do NOT enable the
    # built-in mathjax extension because it inserts its own script tag which
    # can cause a race between the config and the runtime.
    "myst_parser",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid",
]

# Prevent myst_parser from auto-inserting MathJax runtime; we manage MathJax
# inclusion/order manually via html_js_files so config can be applied first.
myst_update_mathjax = False

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
    # Load MathJax config and runtime locally to avoid runtime ordering issues
    "mathjax_config.js",
    # Local loader that will load a vendored runtime (mathjax_runtime.js)
    # when present. Do NOT prefix with `_static/` here; Sphinx will copy the
    # referenced files into the built `_static/` dir and reference them as
    # `_static/<name>`. Using a leading `_static/` causes `_static/_static/...`
    # paths in the output which break file:// viewing.
    "mathjax_local.js",
    # Load Mermaid runtime from our vendored copy in `_static` so docs work
    # even when CDN access is unreliable.
    "mermaid.min.js",
    # Local init file that converts script blocks to .mermaid divs then runs Mermaid
    "mermaid_init.js",
    # Small script to tidy duplicated navbar elements caused by theme options
    "fix_navbar.js",
    # Local MathJax typeset helper — runs MathJax.typeset when the runtime is loaded
    "mathjax_init.js",
    # (Thebe assets removed — Thebe integration disabled to avoid runtime
    # kernel/build attempts and to keep the static site lightweight.)
]

# Thebe configuration for interactive code execution
# Thebe integration has been removed from html_js_files and extensions to
# prevent the client from attempting to start kernels or trigger remote
# builds. If you want to re-enable Thebe in the future, re-add the
# "sphinx_thebe" extension and the corresponding JS entries above.
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
