__author__ = 's'

import os
import json
import logging.config


def setup_logging(
        default_path='logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        error_if_config_missing=False
):
    """

    :param default_path:
    :param default_level:
    :param env_key:
    :param error_if_config_missing:
    :return:
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        if error_if_config_missing:
            raise Exception("Config {} not found")
        logging.basicConfig(level=default_level)
