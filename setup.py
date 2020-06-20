from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='pyviz_med',
    version='0.2',
    description='Holoviews tool for medical image viewing',
    author='Brian Pollack',
    author_email='brianleepollack@gmail.com',
    license='Pitt',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'bokeh>=2.0',
        'holoviews>=1.13.2',
        'datashader',
        'SimpleITK',
        'SimpleITK',
    ],
    zip_safe=False)
