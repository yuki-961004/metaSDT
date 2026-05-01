@echo off
echo =======================================================
echo Syncing C++ core files to R and Python source directories
echo =======================================================

REM 定义源目录
set "SOURCE_CPP=Cpp"

REM 1. 复制到 R 包的 src 目录下
echo [1/2] Copying to R/src/Cpp ...
xcopy "%SOURCE_CPP%" "R\src\Cpp" /E /I /Y /D

REM 2. 复制到 Python 包的 src 目录下
echo [2/2] Copying to Python/src/Cpp ...
xcopy "%SOURCE_CPP%" "Python\src\Cpp" /E /I /Y /D

echo.
echo Sync complete! Press any key to exit.
pause