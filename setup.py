from setuptools import setup, find_packages

setup(name='gaze_predictor_nn',
      version='1.0',
      description='Gaze Predictor Neural Network',
      author='Tobias Niehues',
      author_email='niehues.tobias@gmail.com',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['architectures', 'data'],
      include_package_data=True
)