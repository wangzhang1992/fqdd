import logging
import logging.handlers
import os


def init_logging(filename, log_dir):  #创建文件Logger
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(filename)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, filename+'_log.txt'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
