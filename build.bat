@echo off
echo ===== 贝叶斯优化材料组分系统打包工具 =====
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: 未检测到Python安装。请安装Python 3.9或更高版本。
    pause
    exit /b 1
)

REM 创建并激活虚拟环境
echo 正在创建虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

REM 安装依赖项
echo 正在安装依赖项...
pip install -r requirements.txt
pip install cx_Freeze

REM 创建数据目录（如果不存在）
if not exist data mkdir data

REM 构建应用
echo 正在构建应用...
python setup.py build

REM 复制额外文件
echo 正在复制额外文件...
if not exist build\exe.win-amd64-3.9\data mkdir build\exe.win-amd64-3.9\data

REM 创建安装包
echo 正在创建安装包...
python setup.py bdist_msi

REM 完成
echo.
echo 构建完成！可执行文件位于 build\exe.win-amd64-3.9 目录下
echo 安装包位于 dist 目录下

REM 退出虚拟环境
call venv\Scripts\deactivate.bat

pause