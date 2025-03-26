import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

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

# 定义启动Streamlit的函数
def start_streamlit():
    try:
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
                break
            if "You can now view your Streamlit app in your browser" in line:
                # 服务器已启动，打开浏览器
                webbrowser.open(f"http://localhost:8501")
                break
        
        # 保持进程运行
        process.wait()
    except Exception as e:
        print(f"启动应用时出错: {str(e)}")
        input("按任意键退出...")
        sys.exit(1)

# 在新线程中启动Streamlit
thread = threading.Thread(target=start_streamlit)
thread.daemon = True
thread.start()

# 等待一段时间，如果Streamlit没有成功启动，则自动尝试打开浏览器
time.sleep(5)
webbrowser.open(f"http://localhost:8501")

# 保持主线程运行
while thread.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)