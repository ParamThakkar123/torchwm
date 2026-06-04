from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

README = (
    (HERE / "README.md").read_text(encoding="utf-8")
    if (HERE / "README.md").exists()
    else ""
)

setup(
    name="torchwm",
    version="0.3.2",
    description="A Pytorch Based library for training world models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    packages=find_packages(exclude=("tests", "results", "envs", ".venv", "venv")),
    include_package_data=True,
    entry_points={"console_scripts": ["torchwm=tools.cli:run"]},
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "torchaudio>=2.10.0",
        "einops>=0.8.2",
        "pyyaml>=6.0.3",
        "tqdm>=4.67.1",
        "click>=8.0.0",
    ],
    extras_require={
        "gpu": [
            "torch>=1.13.0",
            "torchvision>=0.14.0",
            "torchaudio>=2.10.0",
        ],
        "gym": [
            "ale-py>=0.11.2",
            "gym>=0.26.2",
            "gymnasium>=1.2.2",
            "mlagents-envs>=0.28.0",
            "pygame>=2.6.1",
            "autorom>=0.6.1",
        ],
        "viz": [
            "fastapi[standard]>=0.116.0",
            "uvicorn>=0.35.0",
            "plotly>=6.5.0",
            "moviepy>=2.2.1",
            "opencv-python>=4.12.0.88",
        ],
        "ml": [
            "tensorboard>=2.20.0",
            "tensorboardx>=2.6.4",
            "wandb>=0.16.0",
            "scikit-learn>=1.8.0",
            "umap-learn>=0.5.11",
        ],
        "dev": [
            "pytest>=9.0.2",
            "pytest-cov>=7.1.0",
            "pre-commit>=4.5.0",
            "mypy>=1.19.1",
        ],
        "docs": [
            "myst-parser>=5.0.0",
            "nbsphinx>=0.9.8",
            "sphinx>=9.1.0",
            "sphinx-autodoc-typehints>=3.6.2",
            "sphinx-copybutton>=0.5.2",
            "sphinx-rtd-theme>=3.1.0",
            "sphinxcontrib-bibtex>=2.6.5",
            "sphinxcontrib-mermaid>=0.9.0",
            "sphinx-thebe>=0.3.0",
            "sphinxext-opengraph>=0.13.0",
            "wheel>=0.46.3",
        ],
        "all": [
            "torchwm[gym,viz,ml,dev,docs]",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="world-models pytorch",
    zip_safe=False,
)
