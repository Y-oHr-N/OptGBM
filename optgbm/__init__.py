"""OptGBM package."""

import logging

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    distribution = get_distribution(__name__)
    __version__ = distribution.version
except DistributionNotFound:
    pass

from lightgbm import *  # noqa

from . import sklearn  # noqa
from . import typing  # noqa
from . import utils  # noqa
from .sklearn import *  # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)

logger.setLevel(logging.INFO)
