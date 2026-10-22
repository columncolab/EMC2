import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="emc2",
    version="1.3.7",
    url="https://github.com/ARM-DOE/EMC2",
    license='MIT',

    author="Robert Jackson",
    author_email="rjackson@anl.gov",

    description="An open source framework for atmospheric model evaluation using observational data",
    long_description=read("README.rst"),
    include_package_data=True,
    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'numpy',
        'xarray',
        'pint',
        'act-atmos',
        'nc-time-axis',
        'openpyxl',
        'matplotlib',
        'scipy',
        'dask',
        'tqdm',
        'netCDF4',
    ],
    
    python_requires='>=3.13,<3.15.1',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
