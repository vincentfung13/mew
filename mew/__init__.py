import logging
import importlib.metadata

from mew.parallel.dist_context import DistContext

__version__ = importlib.metadata.version("mew")
_LOGGER = logging.getLogger("mew")


# Utility log function (only used for training)
def log_info(content, dist_context: DistContext):
    if dist_context.is_main:
        _LOGGER.info(content)
