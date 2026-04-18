import sys
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    """Pick the platform-appropriate requirements file.

    We ship separate pins for Linux (``requirements_linux.txt``) and Windows
    (``requirements_win.txt``); pick based on ``sys.platform`` with Linux as
    the default fallback.
    """
    here = Path(__file__).parent
    req_file = here / ("requirements_win.txt" if sys.platform.startswith("win") else "requirements_linux.txt")
    with open(req_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="e2c",
    version="0.1.0",
    description="Implementation of Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images",
    author="Jared Berry, Ayush Gaggar",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
