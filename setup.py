from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().split("\n")

setup(
    name="pcn",
    version="0.1.0",
    description="Point Cloud Convolutional Networks for tensorflow",
    url="https://github.com/jackd/pcn.git",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    requirements=install_requires,
    zip_safe=True,
)
