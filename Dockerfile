# Use Python 3.10 base image
FROM python:3.10-slim

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install dependencies and NGINX
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge for the appropriate architecture
RUN arch=$(uname -m) && \
    if [ "${arch}" = "x86_64" ]; then \
        wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" -O mambaforge.sh; \
    elif [ "${arch}" = "aarch64" ]; then \
        wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh" -O mambaforge.sh; \
    else \
        echo "Unsupported architecture: ${arch}"; \
        exit 1; \
    fi && \
    bash mambaforge.sh -b -p /opt/mambaforge && \
    rm mambaforge.sh

# Add Mambaforge to PATH
ENV PATH=/opt/mambaforge/bin:$PATH

# Create a new environment with Python 3.10
RUN mamba create -n team2_env python=3.10 -y

# Activate the new environment
SHELL ["mamba", "run", "-n", "team2_env", "/bin/bash", "-c"]

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN mamba install --yes --file /app/requirements.txt && mamba clean --all -f -y

# Copy the NGINX config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the current directory into the container
COPY UI/ .

# Expose NGINX, Streamlit, and Jupyter Notebook ports
EXPOSE 82 5002 6002

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Start NGINX, Streamlit, and Jupyter
CMD service nginx start && \
    streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2 & \
    jupyter notebook --ip=0.0.0.0 --port=6002 --no-browser --allow-root & \
    tail -f /dev/null
