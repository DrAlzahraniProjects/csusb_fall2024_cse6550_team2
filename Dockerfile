# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y wget bzip2 && apt-get clean

# Download and install Miniconda for the appropriate architecture
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O Miniconda.sh; \
    else \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh; \
    fi \
    && bash Miniconda.sh -b -p /opt/miniconda \
    && rm Miniconda.sh

# Add conda to the PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Install Mamba using Miniconda
RUN conda install mamba -n base -c conda-forge

# Create a new environment with Python 3.11
RUN mamba create -n team2_env python=3.11 -y

# Set environment path
ENV PATH="/opt/miniconda/envs/team2_env/bin:$PATH"

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN mamba install --yes --file /app/requirements.txt && mamba clean --all -f -y

# Install Jupyter Notebook
RUN mamba install -c conda-forge jupyter

# Copy the current directory contents into the container
COPY . /app

# Expose ports for Streamlit and Jupyter
EXPOSE 5002
EXPOSE 8888

# Start both Streamlit and Jupyter
CMD ["sh", "-c", "streamlit run app.py --server.port=5002 & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
