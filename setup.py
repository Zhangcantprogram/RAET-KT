from setuptools import setup, find_packages

setup(
    name="raet",
    version="0.1.0",
    description="RAET-KT for multi-concept knowledge tracing with an AKT backbone",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
    ],
)