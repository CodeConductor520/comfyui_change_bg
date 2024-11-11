import logging

class logger:
    def __init__(self, name, log_name, level=logging.INFO):
        # 创建一个logger

        self.logger = logging.getLogger(name)
        self.file_handler = logging.FileHandler(log_name)

        self.logger.setLevel(level)

        # 创建一个文件处理器，并设置级别为DEBUG

        self.file_handler.setLevel(level)

        # 创建一个格式化器，包含时间信息
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 将格式化器添加到处理器
        self.file_handler.setFormatter(formatter)

        # 将处理器添加到logger
        self.logger.addHandler(self.file_handler)

logger_ = logger("REMOVE_BG", "remove_bg_worker.log")

#print("8"*40)

