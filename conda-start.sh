#!/bin/bash

# Set up environment variables
ENV_NAME=$(basename "$PWD")
INSTALL_DIR="$HOME/Miniconda3"
export PATH="$INSTALL_DIR/condabin:$PATH"
COMFY_MANAGER_DIR="$(dirname "$0")/../ComfyUI-Manager"
COMFYUI_DIR="$(dirname "$0")/../.."

# Check if conda environment is available and create it if not
if conda info --envs | grep -i "$ENV_NAME"; then
    echo "$ENV_NAME environment is already available"
else
    echo "$ENV_NAME environment does not exist"
    echo "Creating a new environment"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate environment
source activate "$ENV_NAME"

if [ $? -eq 0 ]; then
    # Install custom node dependencies
    pip install -r requirements.txt

    # Install ComfyUI Manager
    if [ -d "$COMFY_MANAGER_DIR" ]; then
        cd "$COMFY_MANAGER_DIR"
        git pull
    else
        git clone https://github.com/ltdrdata/ComfyUI-Manager "$COMFY_MANAGER_DIR"
        pip install -r "$COMFY_MANAGER_DIR/requirements.txt"
    fi

    # Change the working directory to the root folder
    cd "$COMFYUI_DIR"
    
    # Install packages
    pip install -r requirements.txt
    mim install mmengine mmcv>=2.0.1 mmdet>=3.1.0 mmpose>=1.1.0

    # Install CUDA torch
    pip install torch==2.3.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    
    python main.py --preview-method=taesd --force-fp16 --use-pytorch-cross-attention
else
    echo "Failed to activate environment..."
fi

read -p "Press any key to continue..."
