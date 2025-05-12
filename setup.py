from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="tree-lstm-generator",
    version="0.1.0",
    author="William Wall",
    author_email="",  # Add your email if you want
    description="A tool for visualizing and training Tree-LSTM models for semantic composition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/williamjwall/tree-lstm-generator",
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
            "tree-lstm-train=src.tree_lstm.train:main",
            "tree-lstm-visualize=src.visualization.app:main",
        ],
    },
) 