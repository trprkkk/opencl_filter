@echo off
rem probe_svm.bat — compile+run the SVM capability probe.
setlocal
set MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231
set WINKITS=C:\Program Files (x86)\Windows Kits\10
set WINVER=10.0.26100.0
set SDKROOT=Q:\Downloads\openclfilter\sdk
set "PATH=%MSVC%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%WINKITS%\Include\%WINVER%\shared;%WINKITS%\Include\%WINVER%\ucrt;%WINKITS%\Include\%WINVER%\um;%WINKITS%\Include\%WINVER%\winrt;%SDKROOT%\opencl\include"
set "LIB=%MSVC%\lib\x64;%WINKITS%\Lib\%WINVER%\ucrt\x64;%WINKITS%\Lib\%WINVER%\um\x64;%SDKROOT%\opencl\lib\x64"
cd /d "%~dp0"
if not exist build_host mkdir build_host
cl /nologo /EHsc /MD /std:c++17 /W3 probe_svm.cpp /Fe:build_host\probe_svm.exe /link OpenCL.lib /OUT:build_host\probe_svm.exe
if errorlevel 1 exit /b 1
build_host\probe_svm.exe
endlocal
