import os
import sys
import logging
from datetime import datetime


def logger_init(
    log_filename="monitor", log_level=logging.DEBUG, log_dir="./log/", only_file=False
):
    """
    Args:
        log_filename: 日志文件名.
        log_level: 日志等级.
        log_dir: 日志目录.
        only_file: 是否只保存到日志文件中.
    """
    # 指定日志文件路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filepath = os.path.join(
        log_dir, log_filename + "_" + str(datetime.now())[:10] + ".txt"
    )
    # 指定日志格式
    formatter = "[%(asctime)s] - %(levelname)s: %(message)s"
    # 只保存到日志文件中
    if only_file:
        logging.basicConfig(
            filename=log_filepath,
            level=log_level,
            format=formatter,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    # 保存到日志文件并输出到终端
    else:
        logging.basicConfig(
            level=log_level,
            format=formatter,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler(sys.stdout),
            ],
        )
