#!/bin/bash

# Set up environment variable
ENV_NAME=$(basename "$PWD")

echo "Are you sure you wish to uninstall $ENV_NAME? Close this window if you clicked this by mistake."
read -p "Press any key to continue..."

# Remove conda environment
conda remove --name "$ENV_NAME" --all

echo "Successfully uninstalled the app. Press any key to close."
read -p ""