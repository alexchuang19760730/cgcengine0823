@echo off
REM ============================================================
REM Build CGC fork llama.cpp for Windows (x86_64, MSYS2 MinGW64)
REM ============================================================
REM Prerequisites:
REM   1. MSYS2 installed at C:\msys64
REM   2. MinGW64 toolchain: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
REM ============================================================

set MSYS2=C:\msys64
set MINGW=%MSYS2%\mingw64\bin
set PATH=%MINGW%;%PATH%

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..
set SRC_DIR=%ROOT_DIR%\src\llama.cpp
set BUILD_DIR=%SRC_DIR%\build
set OUT_DIR=%SCRIPT_DIR%
set BUILD_TYPE=Release

echo ==========================================
echo CGC Fork Build (Windows / MinGW64)
echo ==========================================
echo   Source:     %SRC_DIR%
echo   Build dir:  %BUILD_DIR%
echo   Output:     %OUT_DIR%
echo ==========================================

REM Clean if requested
if "%1"=="--rebuild" (
    echo Cleaning previous build...
    rmdir /s /q "%BUILD_DIR%" 2>nul
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Configure
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DGGML_METAL=OFF ^
    -DGGML_VULKAN=OFF ^
    -DGGML_BLAS=OFF ^
    -DGGML_OPENMP=ON ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_EXAMPLES=ON ^
    -DLLAMA_BUILD_SERVER=ON ^
    -DMTP_SUPPORT=ON

REM Build core targets
cmake --build "%BUILD_DIR%" -j8 ^
    --target llama-simple llama-speculative-simple llama-bench llama-server

REM Copy binaries
echo Copying binaries...
copy /y "%BUILD_DIR%\bin\llama-simple.exe" "%OUT_DIR%\llama-simple.exe"
copy /y "%BUILD_DIR%\bin\llama-speculative-simple.exe" "%OUT_DIR%\llama-speculative-simple.exe"
copy /y "%BUILD_DIR%\bin\llama-bench.exe" "%OUT_DIR%\llama-bench.exe"
copy /y "%BUILD_DIR%\bin\llama-server.exe" "%OUT_DIR%\llama-server.exe"

REM Copy required MinGW DLLs
echo Copying MinGW DLLs...
for %%D in (libstdc++-6.dll libwinpthread-1.dll libgomp-1.dll libgcc_s_seh-1.dll) do (
    copy /y "%MINGW%\%%D" "%OUT_DIR%\%%D" 2>nul
)

echo ==========================================
echo Windows build complete!
echo   llama-simple.exe:             %OUT_DIR%\llama-simple.exe
echo   llama-speculative-simple.exe: %OUT_DIR%\llama-speculative-simple.exe
echo   llama-bench.exe:              %OUT_DIR%\llama-bench.exe
echo   llama-server.exe:             %OUT_DIR%\llama-server.exe
echo ==========================================
