@echo off
REM ============================================================
REM Package Windows release: binaries + DLLs + scripts + models
REM ============================================================
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..\..
set OUT_DIR=%SCRIPT_DIR%
set RELEASE_NAME=llama-cpp-cgc-windows-x64
set RELEASE_DIR=%OUT_DIR%\%RELEASE_NAME%
set ZIP_FILE=%OUT_DIR%\%RELEASE_NAME%.zip

echo ==========================================
echo Packaging Windows Release
echo ==========================================

REM Clean previous release
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
if exist "%ZIP_FILE%" del "%ZIP_FILE%"

mkdir "%RELEASE_DIR%"

REM Copy binaries
echo Copying binaries...
for %%F in (llama-simple.exe llama-speculative-simple.exe llama-bench.exe llama-server.exe) do (
    if exist "%OUT_DIR%\%%F" (
        copy /y "%OUT_DIR%\%%F" "%RELEASE_DIR%\%%F"
        echo   + %%F
    ) else (
        echo   ! %%F NOT FOUND (run build-windows.bat first^)
    )
)

REM Copy DLLs
echo Copying DLLs...
for %%D in (libstdc++-6.dll libwinpthread-1.dll libgomp-1.dll libgcc_s_seh-1.dll) do (
    if exist "%OUT_DIR%\%%D" (
        copy /y "%OUT_DIR%\%%D" "%RELEASE_DIR%\%%D"
        echo   + %%D
    ) else (
        REM Try MSYS2
        if exist "C:\msys64\mingw64\bin\%%D" (
            copy /y "C:\msys64\mingw64\bin\%%D" "%RELEASE_DIR%\%%D"
            echo   + %%D (from MSYS2^)
        ) else (
            echo   ! %%D NOT FOUND
        )
    )
)

REM Copy scripts
echo Copying scripts...
copy /y "%OUT_DIR%\run-windows.sh" "%RELEASE_DIR%\run.sh"
copy /y "%OUT_DIR%\benchmark-windows.sh" "%RELEASE_DIR%\benchmark.sh"
echo   + run.sh
echo   + benchmark.sh

REM Copy README
echo # CGC llama.cpp Windows Release> "%RELEASE_DIR%\README.md"
echo.>> "%RELEASE_DIR%\README.md"
echo ## Quick Start>> "%RELEASE_DIR%\README.md"
echo.>> "%RELEASE_DIR%\README.md"
echo ```>> "%RELEASE_DIR%\README.md"
echo REM Run (basic^)>> "%RELEASE_DIR%\README.md"
echo .\llama-simple.exe -m model.gguf -ngl 4 -c 2048>> "%RELEASE_DIR%\README.md"
echo.>> "%RELEASE_DIR%\README.md"
echo REM Run (MTP^)>> "%RELEASE_DIR%\README.md"
echo .\llama-speculative-simple.exe -m model.gguf --spec-type draft-mtp --spec-draft-n-max 2 -ngl 4>> "%RELEASE_DIR%\README.md"
echo.>> "%RELEASE_DIR%\README.md"
echo REM Run (server^)>> "%RELEASE_DIR%\README.md"
echo .\llama-server.exe -m model.gguf --host 0.0.0.0 --port 1234 -ngl 4>> "%RELEASE_DIR%\README.md"
echo ```>> "%RELEASE_DIR%\README.md"
echo   + README.md

REM Create zip
echo.
echo Creating zip...
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath '%ZIP_FILE%' -Force"
echo   Created: %ZIP_FILE%

REM Summary
echo.
echo ==========================================
echo Release packaged!
echo   Dir:  %RELEASE_DIR%
echo   Zip:  %ZIP_FILE%
echo ==========================================
dir "%RELEASE_DIR%"
