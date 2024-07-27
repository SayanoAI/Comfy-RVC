@echo off

REM set up env variables
for /f %%i in ('cd') do set ENV_NAME=%%~nxi
SET INSTALL_DIR=%userprofile%\Miniconda3
SET PATH=%INSTALL_DIR%\condabin;%PATH%
SET COMFY_MANAGER_DIR=%~dp0..\ComfyUI-Manager
FOR %%I IN (%~dp0..\..) DO SET COMFYUI_DIR=%%~fI

REM This script will check if a conda environment is available and create it if not	
CALL conda info --envs | findstr /i %ENV_NAME%
if %errorlevel% == 0 (
    echo %ENV_NAME% environment is already available
) else (
    echo %ENV_NAME% environment does not exist
    echo Creating a new environment
    CALL conda create -n %ENV_NAME% python=3.10 -y
)

rem Activate environment
CALL conda activate %ENV_NAME%

if %errorlevel% == 0 (

	@REM Install custom node dependencies
	CALL pip install -r requirements.txt

	@REM install comfyui manager
	if exist  %COMFY_MANAGER_DIR% (
		CD %COMFY_MANAGER_DIR%
		CALL git pull
	) else (
		CALL git clone https://github.com/ltdrdata/ComfyUI-Manager %COMFY_MANAGER_DIR%
		CALL pip install -r %COMFY_MANAGER_DIR%\requirements.txt
	)

	REM Change the working directory to the root folder
	CD %COMFYUI_DIR%
    
	rem install packages
    CALL pip install -r requirements.txt

	rem install CUDA torch
	CALL pip install torch==2.0.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
	
    CALL python main.py --disable-smart-memory
) else (
    echo Failed to activate environment...
)
PAUSE