# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y wget

# Determine system architecture and we will install the corresponding version of Miniconda
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash Miniconda3-latest-Linux-*.sh -b -p /root/miniconda3 && \
    rm Miniconda3-latest-Linux-*.sh && \
    apt-get clean

# Set the Mamba root prefix and add conda to the PATH
ENV PATH="/root/miniconda3/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/root/miniconda3"

# Install Mamba using Conda and create a new environment with Python 3.11
RUN conda install mamba -c conda-forge -y && \
    mamba create -n team2_env python=3.11 -y && \
    mamba clean --all -f -y

# Set the environment path to use team2_env and ensure bash is used
ENV PATH="/root/miniconda3/envs/team2_env/bin:$PATH"

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN mamba install --yes --file /app/requirements.txt && \
    mamba clean --all -f -y

# Install Jupyter Notebook
RUN mamba install -c conda-forge jupyter

# Install NGINX
RUN apt-get update && apt-get install -y nginx

# Copy NGINX config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the current directory contents into the container
COPY UI/ .

# Expose ports for NGINX, Streamlit, and Jupyter
EXPOSE 82
EXPOSE 5002
EXPOSE 6002

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Start NGINX, Streamlit, and Jupyter
CMD service nginx start && \
    streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2 & \
    jupyter notebook --ip=0.0.0.0 --port=6002 --no-browser --allow-root & \
    tail -f /dev/null
