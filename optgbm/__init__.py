"""OptGBM package."""

__version__ = '0.0.0'

try:
    from .sklearn import *  # noqa
except ImportError:
    pass
