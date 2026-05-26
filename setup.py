from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eemd-ica-factor-analysis",
    version="0.1.0",
    author="EEMD-ICA Contributors",
    description="Factor analysis of financial time series using EEMD-ICA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/eemd-ica-factor-analysis",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "plot": ["matplotlib>=3.4"],
        "stats": ["statsmodels>=0.13"],
        "dev": ["pytest>=7.0", "matplotlib>=3.4", "statsmodels>=0.13"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
)
