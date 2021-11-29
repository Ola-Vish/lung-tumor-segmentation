#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='lung-tumor-segmentation',
    version='0.0.0',
    description='Lung tumor segmentation project',
    author='Olga',
    author_email='olgavish1@gmail.com',
    url='https://github.com/Ola-Vish/lung-tumor-segmentation',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

