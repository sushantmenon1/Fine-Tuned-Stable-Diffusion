import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
from pathlib import Path

class PostInstall(install):
    def run(self):
        install.run(self)
        download_models()

def download_models():
    # create model directory
    model_dir = Path(get_python_lib()).joinpath('generate').joinpath('model')
    model_dir.mkdir(parents=True, exist_ok=True)

    # download weights
    subprocess.run(['curl', 
                    'https://storage.googleapis.com/playground-sushant-eefk/custom%20model/pytorch_custom_diffusion_weights.bin', 
                    '-o', 
                    model_dir.joinpath('pytorch_custom_diffusion_weights.bin')])

    # download encodings
    subprocess.run(['curl', 
                    'https://storage.googleapis.com/playground-sushant-eefk/custom%20model/kerala.bin', 
                    '-o', 
                    model_dir.joinpath('kerala.bin')])

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
      cmdclass={'install': PostInstall},
      entry_points={'console_scripts': ['generate = generate.main:generate']})
