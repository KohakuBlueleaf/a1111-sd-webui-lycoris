import logging

__all__ = ['logger']

logger = logging.getLogger("lycoris")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)