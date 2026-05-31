"""Sphinx configuration for TorchWM documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "TorchWM"
copyright = f"{datetime.now().year}, Param Thakkar"
author = "Param Thakkar"

# Auto-read version from world_models package
import world_models

release = world_models.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
]

# Let MyST/Sphinx emit math nodes and let sphinx.ext.mathjax load MathJax in a
# deterministic order. This is more reliable than hand-loading a placeholder
# runtime from _static and supports both ```{math}``` fences and $...$ spans.
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "packages": {"[+]": ["ams"]},
    },
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre", "code"],
    },
}

# Heavy optional runtimes are mocked so the API reference can build in a docs
# environment without installing all simulation backends.
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
    "sklearn",
    "sklearn.manifold",
    "h5py",
    "huggingface_hub",
    "matplotlib",
    "matplotlib.pyplot",
    "umap",
    "torch",
    "torch.nn",
    "torch._C",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/paramthakkar123/torchwm",
    "navigation_depth": 2,
    "show_nav_level": 1,
    "navbar_end": ["navbar-icon-links", "search-field"],
}

# sphinxcontrib-mermaid emits raw Mermaid blocks and we render them client-side
# with the vendored Mermaid runtime plus _static/mermaid_init.js.
mermaid_output_format = "raw"
mermaid_version = "10.9.1"
mermaid_init_js = "mermaid.initialize({startOnLoad:false, securityLevel:'loose'});"

html_js_files = [
    "mermaid.min.js",
    "mermaid_init.js",
    "fix_navbar.js",
]

# Copy both our _static assets and the `images/` dir so images referenced from
# pages are available at build time (placed under _static/).
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
