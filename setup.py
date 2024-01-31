import os
from setuptools import setup, find_packages

with open(os.path.join("requirements", "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open(os.path.join("requirements", "test_requirements.txt")) as f:
    test_requirements = f.read().splitlines()

setup(
    name="fedless",
    packages=find_packages(),
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={"dev": test_requirements},
    python_requires=">=3.7",
    entry_points="""
        [console_scripts]
        fedless=fedless.controller.fedless:cli
    """,
)
