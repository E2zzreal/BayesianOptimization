import sys
from cx_Freeze import setup, Executable

# 依赖项
dependencies = [
    "sklearn",  # 修改：scikit-learn 的导入名称是 sklearn
    "xgboost",  # 正确
    "streamlit",  # 正确
    "skopt",  # 正确：scikit-optimize 的导入名称是 skopt
    "GPy",  # 正确
    "GPyOpt",  # 正确
    "joblib",  # 正确
    "pandas",  # 正确
    "numpy"  # 正确
]

# 构建选项
build_exe_options = {
    "packages": dependencies + ["os", "sys", "scipy"],  # 移除重复的 sklearn 和 numpy
    "excludes": ["tkinter", "GPy.testing", "*.tests"],
    "include_files": [("app", "app"), ("requirements.txt", "requirements.txt"), "data/"],
    "optimize": 2,
    "bin_includes": [],  # 添加这一行，避免复制系统文件
    "bin_excludes": ["VCRUNTIME140.dll", "MSVCP140.dll"],  # 排除一些系统DLL
    "include_msvcr": True,  # 包含MSVC运行时
}

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
    options={"build_exe": build_exe_options},
    executables=executables
)