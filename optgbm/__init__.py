"""OptGBM package."""

__version__ = '0.0.0'

try:
    __SETUP__  # type: ignore
except NameError:
    __SETUP__ = False  # type: ignore

if not __SETUP__:
    from .sklearn import *  # noqa
