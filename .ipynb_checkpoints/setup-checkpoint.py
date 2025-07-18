from setuptools import setup, find_packages

setup(
    name="hybrid_threat_detection",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas==2.3.1",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
    ],
)