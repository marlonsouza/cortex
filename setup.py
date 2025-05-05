"""Setup configuration for the Cortex package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="cortex",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered project scaffolding tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cortex",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cortex=cortex.agent:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest==8.0.0",
            "black==24.1.1",
            "isort==5.13.2",
            "flake8==7.0.0",
            "flake8-docstrings==1.7.0",
            "pre-commit==3.5.0",
        ],
    },
)
