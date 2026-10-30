@echo off
rem build_host_deblock.bat — plain cl.exe build of the sixth host plugin
rem (KDeblockOCL AviSynth+ DLL, KFM/Deblock.cu port) + its headless
rem verification exe, using the verified MSVC x64 recipe (same as
rem build_host_edgelevel.bat / build_host_temporalnr.bat / build_host_deband.bat /
rem build_host_mergestatic.bat / build_host_noiseclip.bat). Run from
rem repo\src\host\plugin.
rem
rem Prereqs on this machine (see docs note + task recipe):
rem   MSVC 14.51 (VS 18 Community), Windows Kits 10.0.26100.0,
rem   OpenCL headers at Q:\Downloads\openclfilter\sdk\opencl\include,
rem   OpenCL.lib (generated from System32 OpenCL.dll exports) at
rem   Q:\Downloads\openclfilter\sdk\opencl\lib\x64,
rem   AviSynth+ SDK headers at Q:\Downloads\openclfilter\sdk.

setlocal enabledelayedexpansion

set MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231
set WINKITS=C:\Program Files (x86)\Windows Kits\10
set WINVER=10.0.26100.0
set ROOT=%~dp0..\..\..
set SDKROOT=Q:\Downloads\openclfilter\sdk
set OUTDIR=%~dp0build_host

set "PATH=%MSVC%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%WINKITS%\Include\%WINVER%\shared;%WINKITS%\Include\%WINVER%\ucrt;%WINKITS%\Include\%WINVER%\um;%WINKITS%\Include\%WINVER%\winrt;%SDKROOT%\opencl\include"
set "LIB=%MSVC%\lib\x64;%WINKITS%\Lib\%WINVER%\ucrt\x64;%WINKITS%\Lib\%WINVER%\um\x64;%SDKROOT%\opencl\lib\x64"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

rem 1. embed the kernel source into the DLL (self-contained plugin)
python3 "%~dp0embed_cl.py" "%ROOT%\src\opencl\kfm\kernels\kfm_deblock.cl" "%OUTDIR%\kfm_deblock_cl.h" kfm_deblock_cl_src
if errorlevel 1 exit /b 1
copy /y "%OUTDIR%\kfm_deblock_cl.h" "%~dp0kfm_deblock_cl.h" >nul

rem 2. plugin DLL (autodetected by AviSynth+ via AvisynthPluginInit3)
cl /nologo /EHsc /MD /std:c++17 /W3 /LD ^
   kfm_deblock_avs.cpp "%ROOT%\src\host\ocl_host.cpp" ^
   /I"%ROOT%\src\host" /I"%SDKROOT%" ^
   /Fe"%OUTDIR%\kfm_deblock_avs.dll" ^
   /link OpenCL.lib /OUT:"%OUTDIR%\kfm_deblock_avs.dll"
if errorlevel 1 exit /b 1

rem 3. headless verification exe
cl /nologo /EHsc /MD /std:c++17 /W3 ^
   test_kfm_deblock.cpp ^
   /I"%SDKROOT%" /I"%~dp0." ^
   /Fe"%OUTDIR%\test_kfm_deblock.exe" ^
   /link /OUT:"%OUTDIR%\test_kfm_deblock.exe"
if errorlevel 1 exit /b 1

echo.
echo build OK:
dir /b "%OUTDIR%\kfm_deblock_avs.dll" "%OUTDIR%\test_kfm_deblock.exe"
endlocal
