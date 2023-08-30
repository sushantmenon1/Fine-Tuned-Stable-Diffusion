from setuptools import setup

setup(
  name='myapp', 
  version='1.0',
  py_modules=['Application'], 
  entry_points={
    'console_scripts': ['generate=Application.main:generate']
  }
)
