# Use Python as the base image
FROM python:3.8

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge for the appropriate architecture
RUN arch=$(uname -m) && \
    if [ "${arch}" = "x86_64" ]; then \
        wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh; \
    elif [ "${arch}" = "aarch64" ]; then \
        wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" -O miniforge.sh; \
    else \
        echo "Unsupported architecture: ${arch}"; \
        exit 1; \
    fi && \
    bash miniforge.sh -b -p /opt/miniforge && \
    rm miniforge.sh

# Add Mambaforge to PATH
ENV PATH=/opt/miniforge/bin:$PATH

# Create a new environment with Python 3.8
RUN mamba create -n team2_env python=3.8 -y

# Activate the new environment
SHELL ["mamba", "run", "-n", "team2_env", "/bin/bash", "-c"]

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN mamba install --yes --file requirements.txt && mamba clean --all -f -y
# Install Rust and Cargo
RUN apt-get update && apt-get install -y \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y \
    && source $HOME/.cargo/env

# Add Rust to the PATH
ENV PATH=/root/.cargo/bin:$PATH
# Install additional Python packages (including NeMo Guardrails)
RUN pip install nemoguardrails pymilvus[model] langchain langchain_community langchain_huggingface langchain_milvus beautifulsoup4 requests nltk langchain_mistralai sentence-transformers

# Set environment variables for Clang
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

# Install C++ runtime dependencies (specific to NeMo Guardrails)
RUN apt-get update && apt-get install -y \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "import nemoguardrails; print('NeMo Guardrails installed')"
# Define a named volume for persistent data storage
# VOLUME /app/data

# Copy the current directory contents into the container at /app
COPY . /app

# Copy only the Jupyter notebook to the /Jupyter directory
COPY Jupyter /Jupyter

# Set the Streamlit ENV for configuration
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Streamlit port
EXPOSE 5002
# Jupyter Notebook port
EXPOSE 6002

# Add the conda environment's bin directory to PATH
ENV PATH=/opt/miniforge/envs/team2_env/bin:$PATH

# Run the Streamlit app and Jupyter
CMD ["sh", "-c", "streamlit run app/main.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2 & jupyter notebook --ip=0.0.0.0 --port=6002 --no-browser --allow-root --NotebookApp.base_url=/team2/jupyter --NotebookApp.token='' --notebook-dir=/Jupyter/"]
