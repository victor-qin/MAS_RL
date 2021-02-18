from setuptools import setup, find_packages

setup(name='Quadcopter_Simcon',
      version='0.1.1',
      install_requires=['gym', 'numpy', 'matplotlib'],
      test_requires=["pytest", "mock"],
      packages=find_packages()
      )
