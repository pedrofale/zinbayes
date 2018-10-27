from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ZINBayes",
    version="0.1",
    author="Pedro F. Ferreira",
    description="Bayesian Zero-Inflated Negative Binomial Factorization of single-cell RNA-seq data.",
    packages=find_packages(),
    install_requires=requirements,
)
