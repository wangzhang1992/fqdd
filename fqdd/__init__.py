""" Comprehensive speech processing toolkit
"""
import os
from . import models  # noqa
from . import utils  # noqa
from . import nnets  # noqa
from . import prepare_data  # noqa
from . import lm  # noqa
from . import asr 

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version

'''
asr
drwxr-xr-x 2 root root   64 Apr 30  2021 lm
drwxr-xr-x 2 root root   64 Dec 20 10:24 models
drwxr-xr-x 2 root root   64 Dec 16 13:48 nnets
drwxr-xr-x 2 root root   64 Dec 20 11:50 prepare_data
drwxr-xr-x 2 root root   64 Jun 15  2021 test
-rw-r--r-- 1 root root 7.4K Jun 23 13:23 tmp.py
drwxr-xr-x 2 root root   64 Dec 20 11:32 utils
'''
