from setuptools import setup, find_packages
setup(
    name='chaotic_mtd_ids',
    version='1.0',
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
)