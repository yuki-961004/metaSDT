@echo off
echo =======================================================
echo Syncing C++ core files to R and Python source directories
echo =======================================================

REM 定义源目录
set "SOURCE_CPP=Cpp"

REM 1. 复制到 R 包的 src 和 inst/include 目录下
echo [1/3] Copying to R/src/Cpp ...
xcopy "%SOURCE_CPP%" "R\src\Cpp" /E /I /Y /D
echo [2/3] Copying headers to R/inst/include ...
xcopy "%SOURCE_CPP%\include" "R\inst\include" /E /I /Y /D

REM 2. 复制到 Python 包的 src 目录下
echo [3/3] Copying to Python/src/Cpp ...
xcopy "%SOURCE_CPP%" "Python\src\Cpp" /E /I /Y /D

echo.
echo Sync complete! Press any key to exit.
pause