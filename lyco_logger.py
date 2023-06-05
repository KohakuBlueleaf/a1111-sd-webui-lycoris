import logging
from modules import shared

__all__ = ['logger']

logger = logging.getLogger("lycoris")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if shared.cmd_opts.lyco_debug:
    logger.setLevel(logging.DEBUG)