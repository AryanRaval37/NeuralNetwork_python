from setuptools import setup, find_packages

setup(
    name="neuralnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "tqdm",
    ],
    author="Aryan Kraval",
    description="A simple neural network library",
)
