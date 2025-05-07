from setuptools import setup, find_packages

setup(
    name="trader",
    version="0.1.0",
    packages=find_packages(include=["trader", "trader.*"]),
)
