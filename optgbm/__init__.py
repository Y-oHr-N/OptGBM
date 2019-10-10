"""OptGBM package."""

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    distribution = get_distribution(__name__)
    __version__ = distribution.version
except DistributionNotFound:
    pass

from . import sklearn  # noqa
from . import utils  # noqa
