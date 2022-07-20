#! /usr/bin/env python

"""
@author: maedbhking
based heavily on flexible functionality of nilearn `setup.py`
"""

descr = """A python package for cortical neuroimaging..."""

import sys
import os

from setuptools import setup, find_packages

def load_version():
    """Executes imageUtils/version.py in a globals dictionary and return it.
    Note: importing imageUtils is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('imageUtils', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


def list_required_packages():
    required_packages = []
    required_packages_orig = ['%s>=%s' % (mod, meta['min_version'])
                              for mod, meta
                              in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
                              ]
    for package in required_packages_orig:
        required_packages.append(package)
    return required_packages


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'imageUtils'
DESCRIPTION = 'Mapping and plotting cerebellar and cortical fMRI data in Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Maedbh King'
MAINTAINER_EMAIL = 'maedbhking@gmail.com'
URL = 'https://github.com/maedbhking/imageUtils'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/maedbhking/imageUtils/archive/refs/tags/v.1.1.1.tar.gz'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_imageUtils_installing=True)

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ],
          packages=find_packages(),
          package_data={
              'imageUtils': ['surfaces/*.surf.gii', 'surfaces/*dscalar.nii', 'surfaces/*.border', 'surfaces/*.dlabel.nii'],
          },
          install_requires=list_required_packages(),
          python_requires='>=3.6',
          )