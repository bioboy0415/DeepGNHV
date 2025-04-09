# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# This section of code is adapted from the Swin Transformer project
# Source: https://github.com/microsoft/Swin-Transformer


import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir, log_type="Train", name=""):
    # Define log message format
    log_format = {
        'fmt': '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s',
        'color_fmt': (
            colored('[%(asctime)s %(name)s]', 'green')
            + colored('(%(filename)s %(lineno)d)', 'yellow')
            + ': %(levelname)s %(message)s'
        ),
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

    # Create and configure a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=log_format['color_fmt'], datefmt=log_format['datefmt']))
    logger.addHandler(console_handler)

    # Create a file handler
    log_file = os.path.join(output_dir, f'{log_type}_logger_{name}.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=log_format['fmt'], datefmt=log_format['datefmt']))
    logger.addHandler(file_handler)

    return logger
