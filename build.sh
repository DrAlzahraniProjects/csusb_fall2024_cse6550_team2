#!/bin/bash

Exit immediately if a command exits with a non-zero status
set -e

Update package list and install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
  wget \
  bzip2 \
  ca-certificates \
  build-essential \
  python3-dev

Install Mambaforge
echo "Installing Mambaforge..."
wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh
bash miniforge.sh -b -p /opt/miniforge
rm miniforge.sh

Add Mambaforge to PATH
export PATH=/opt/miniforge/bin:$PATH

Create and activate the Conda environment
echo "Setting up Conda environment..."
mamba create -n team2_env python=3.10 -y
source activate team2_env

Install Python dependencies
echo "Installing Python dependencies..."
mamba install --yes --file requirements.txt
pip install nemoguardrails pymilvus[model] langchain beautifulsoup4 requests nltk sentence-transformers scipy

Set Streamlit environment variables
echo "Setting Streamlit environment variables..."
export STREAMLIT_SERVER_BASEURLPATH="/team2"
export STREAMLIT_SERVER_PORT=5002

echo "Build script completed successfully."
