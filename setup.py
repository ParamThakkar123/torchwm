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
    version="0.4.1",
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
        "einops>=0.8.2",
        "pyyaml>=6.0.3",
        "tqdm>=4.67.1",
        "requests>=2.32.0",
        "click>=8.0.0",
    ],
    extras_require={
        "gym": [
            "gym>=0.26.2",
            "gymnasium>=1.2.2",
            "opencv-python>=4.12.0.88",
            "moviepy>=2.2.1",
            "pygame>=2.6.1",
        ],
        "dmc": [
            "dm-control>=1.0.28",
            "mujoco>=3.3.0",
            "gymnasium>=1.2.2",
            "opencv-python>=4.12.0.88",
            "moviepy>=2.2.1",
        ],
        "atari": [
            "ale-py>=0.11.2",
            "autorom>=0.6.1",
            "gymnasium>=1.2.2",
            "opencv-python>=4.12.0.88",
        ],
        "datasets": [
            "h5py>=3.11.0",
            "huggingface-hub>=1.14.0",
        ],
        "ml-agents": [
            "mlagents-envs>=0.28.0",
        ],
        "mujoco": [
            "mujoco>=3.2.0",
            "gymnasium>=1.2.2",
        ],
        "robotics": [
            "gymnasium-robotics>=1.3.0",
        ],
        "brax": [
            "brax>=0.13.0",
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
            "playwright>=1.60.0",
            "selenium>=4.41.0",
        ],
        "docs": [
            "myst-parser>=2.0.0,<5.0.0",
            "sphinx>=8.0.0,<10.0.0",
            "sphinx-autodoc-typehints<1.26.0",
            "sphinx-copybutton>=0.5.2",
            "sphinx-rtd-theme>=3.1.0",
            "pydata-sphinx-theme>=0.16.0",
            "sphinxcontrib-bibtex>=2.6.5",
            "sphinxcontrib-mermaid>=0.9.0",
            "sphinx-thebe>=0.3.0",
            "sphinxext-opengraph>=0.13.0",
        ],
        "all": [
            "torchwm[gym,dmc,atari,datasets,viz,ml,ml-agents,mujoco,robotics,brax,dev,docs]",
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
