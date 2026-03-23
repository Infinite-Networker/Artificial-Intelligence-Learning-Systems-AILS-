"""
AILS Package Setup Configuration
Created by Cherry Computer Ltd.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ails-framework",
    version="1.0.0",
    author="Cherry Computer Ltd.",
    author_email="contact@cherrycomputer.ltd",
    description=(
        "Artificial Intelligence Learning System — "
        "A comprehensive autonomous AI framework"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CherryComputerLtd/AILS",
    project_urls={
        "Bug Reports": "https://github.com/CherryComputerLtd/AILS/issues",
        "Documentation": "https://github.com/CherryComputerLtd/AILS/wiki",
        "Source": "https://github.com/CherryComputerLtd/AILS",
        "Company": "https://github.com/CherryComputerLtd",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "vision": [
            "opencv-python>=4.8.0",
            "Pillow>=10.0.0",
        ],
        "nlp": [
            "transformers>=4.30.0",
            "nltk>=3.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=(
        "artificial-intelligence machine-learning deep-learning nlp "
        "computer-vision reinforcement-learning ethics autonomous-learning"
    ),
    entry_points={
        "console_scripts": [
            "ails=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
