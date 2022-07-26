
"""
Helper functions to download Cortical NeuroImaging datasets
"""
import gzip
import os
import pkg_resources
import warnings

from distutils.version import LooseVersion

from .version import _check_module_dependencies, __version__

from .atlas import (fetch_yeo_2011
                    )

def _nibabel2_deprecation_warning():
    msg = ('Support for Nibabel 2.x is deprecated and will stop '
           'in release 0.9.0. Please consider upgrading to '
           'Nibabel 3.x.')
    warnings.filterwarnings('once', message=msg)
    warnings.warn(message=msg,
                  category=FutureWarning,
                  stacklevel=3)

def _nibabel_deprecation_warnings():
    """Give a deprecation warning is the version of
    Nibabel is < 3.0.0.
    """
    # Nibabel should be installed or we would
    # have had an error when calling
    # _check_module_dependencies
    dist = pkg_resources.get_distribution('nibabel')
    nib_version = LooseVersion(dist.version)
    if nib_version < '3.0':
        _nibabel2_deprecation_warning()

_check_module_dependencies()
_nibabel_deprecation_warnings()

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

__all__ = [fetch_yeo_2011]