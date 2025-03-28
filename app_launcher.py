import os
import sys
import subprocess
import webbrowser
import time
import threading
import logging
from pathlib import Path
from datetime import datetime

# 设置应用根目录
app_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_dir)

# 设置环境变量
os.environ["PYTHONPATH"] = app_dir
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# 创建数据目录（如果不存在）
data_dir = os.path.join(app_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 创建日志目录（如果不存在）
logs_dir = os.path.join(app_dir, "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 设置日志记录
def setup_logging():
    """配置日志系统"""
    # 创建日志文件名（包含日期）
    log_filename = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 配置日志格式
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(log_format)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器（避免重复）
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 定义启动Streamlit的函数
def start_streamlit():
    logger = setup_logging()
    try:
        logger.info("正在启动Streamlit应用...")
        # 使用subprocess启动Streamlit
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", os.path.join(app_dir, "app", "main.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 监控输出以确定服务器何时启动
        while True:
            line = process.stdout.readline()
            if not line:
                logger.warning("Streamlit进程输出结束，可能未正常启动")
                break
            if "You can now view your Streamlit app in your browser" in line:
                # 服务器已启动，打开浏览器
                logger.info("Streamlit服务器已成功启动")
                webbrowser.open(f"http://localhost:8501")
                break
        
        # 保持进程运行
        process.wait()
    except Exception as e:
        logger.error(f"启动应用时出错: {str(e)}", exc_info=True)
        input("按任意键退出...")
        sys.exit(1)

# 初始化日志系统
logger = setup_logging()
logger.info("应用启动中...")

# 在新线程中启动Streamlit
thread = threading.Thread(target=start_streamlit)
thread.daemon = True
thread.start()
logger.info("Streamlit线程已启动")

# 等待一段时间，如果Streamlit没有成功启动，则自动尝试打开浏览器
time.sleep(5)
logger.info("尝试打开浏览器访问应用")
webbrowser.open(f"http://localhost:8501")

# 保持主线程运行
while thread.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号，应用正在退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"主线程发生异常: {str(e)}", exc_info=True)
        sys.exit(1)