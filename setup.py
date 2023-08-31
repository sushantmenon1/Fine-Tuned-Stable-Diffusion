import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
from pathlib import Path

setup(name='generate', 
      version='0.1.0',
      author='Sushant Menon',
      author_email = "sushantmenon1@gmail.com",
      install_requires=["torch", 
                        "numpy",
                        "diffusers @ git+https://github.com/sushantmenon1/diffusers.git",
                        'transformers',
                        'accelerate',
                        'opencv-python',
                        'Pillow'],
      packages=find_packages(include=['generate', 'generate.*']),
      py_modules=['generate'],
      entry_points={'console_scripts': ['generate = generate.main:generate']})
