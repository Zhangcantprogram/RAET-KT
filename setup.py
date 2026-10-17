from setuptools import setup, find_packages

setup(
    name="rapl",
    version="0.1.0",
    description="Minimal AKT-only implementation based on pyKT toolkit",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
    ],
)