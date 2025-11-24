from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
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
