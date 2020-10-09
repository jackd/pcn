import os

from kblocks.gin_utils.config import try_register_config_dir

PCN_CONFIG_DIR = os.path.realpath(os.path.dirname(__file__))
try_register_config_dir("PCN_CONFIG", PCN_CONFIG_DIR)
