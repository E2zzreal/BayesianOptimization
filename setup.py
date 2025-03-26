import sys
from cx_Freeze import setup, Executable

# 依赖项
dependencies = [
    "scikit-learn",
    "xgboost",
    "streamlit",
    "scikit-optimize",
    "GPy",
    "GPyOpt",
    "joblib",
    "pandas",
    "numpy"
]

# 构建选项
build_options = {
    "packages": dependencies,
    "excludes": [],
    "include_files": [("app", "app"), ("requirements.txt", "requirements.txt")],
}

# 可执行文件选项
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # 使用Windows GUI模式，不显示控制台

executables = [
    Executable(
        "app_launcher.py",  # 启动脚本
        base=base,
        target_name="贝叶斯优化材料组分系统.exe",
        icon="app_icon.ico",  # 应用图标
        shortcut_name="贝叶斯优化材料组分系统",
        shortcut_dir="DesktopFolder",
    )
]

# 设置信息
setup(
    name="BayesianOptimizationSystem",
    version="1.0.0",
    description="贝叶斯优化材料组分系统",
    options={"build_exe": build_options},
    executables=executables
)