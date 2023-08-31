from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
from pathlib import Path

class PostInstall(install):
    def run(self):
        install.run(self)
        download_models()

def download_models():
    from google.cloud import storage
    from google.cloud.storage.blob import Blob

    # GCP bucket details
    bucket_name = Path('playground-sushant-eefk')
    prefix = Path('custom model')
    blobs = ['kerala.bin', 'pytorch_custom_diffusion_weights.bin']
    
    # create model directory
    model_dir = Path(get_python_lib()).joinpath('generate').joinpath('model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize
    storage_client = storage.Client()
    bucket = storage_client.bucket(str(bucket_name))
    blob_names = [prefix.joinpath('kerala.bin'),
                  prefix.joinpath('pytorch_custom_diffusion_weights.bin')]
    
    # download
    for blob_name in blob_names:
        blob = Blob(str(blob_name), bucket)
        file_name = blob.name.split("/")[-1]
        print('Starting download')
        print(model_dir)
        blob.download_to_filename(model_dir.joinpath(file_name)) 

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
                        'Pillow', 
                        'google-cloud-storage'],
      packages=find_packages(include=['generate', 'generate.*']),
      py_modules=['generate'],
      cmdclass={'install': PostInstall},
      entry_points={'console_scripts': ['generate = generate.main:generate']})
