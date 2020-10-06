import os

from absl import logging

PCN_CONFIG = os.path.realpath(os.path.dirname(__file__))

if os.environ.get("PCN_CONFIG"):
    logging.warning("PCN_CONFIG environment variable already set.")
else:
    os.environ["PCN_CONFIG"] = PCN_CONFIG
