import os
from logging.config import fileConfig
curr_dir = os.path.abspath(os.path.dirname(__file__))
fileConfig(
    os.path.join(curr_dir, 'logging.conf'),
    disable_existing_loggers=False,
)