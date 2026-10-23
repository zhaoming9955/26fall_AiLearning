""" 
提供项目通用工具
当前负责创建日志记录器, 将运行信息输出到终端和日志文件
"""

# logging 是 python 标准库提供的日志模块, 无需额外安装
import logging
from pathlib import Path

def setup_logger(log_dir: Path) -> logging.Logger:
    """ 创建日志目录, 配置终端与文件输出, 返回日志记录器 """
    
    # 创建日志目录, 目录已经存在时继续使用
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取指定名称的记录器, INFO 级别用于记录正常运行信息
    logger = logging.getLogger("fashionmnist")
    logger.setLevel(logging.INFO)
    
    # 禁止继续向父记录器传递, 避免同一条信息重复输出
    logger.propagate = False
    
    # 如果重复调用本函数, 先关闭旧的输出处理器
    # [:] 创建列表副本，避免在遍历原列表时修改它。
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # 定义每条日志的格式：时间、级别、具体消息。
    # %(...)s 是 logging 使用的占位符，由日志模块填入对应字段。
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器：以 UTF-8 编码保存中文日志。
    # mode="a" 表示追加，保留之前运行留下的记录。
    file_handler = logging.FileHandler(
        filename=log_dir / "train.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # 终端处理器：将相同的信息显示在终端。
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 一个记录器连接两个处理器，实现同时输出。
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger