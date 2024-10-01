# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install wget, NGINX, and Miniconda based on system architecture
RUN apt-get update && apt-get install -y wget nginx && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash Miniconda3-latest-Linux-*.sh -b -p /root/miniconda3 && \
    rm Miniconda3-latest-Linux-*.sh && \
    apt-get remove -y wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the Mamba root prefix and add conda to the PATH
ENV PATH="/root/miniconda3/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/root/miniconda3"

COPY requirements.txt /app/requirements.txt

# Install Mamba and create a new environment with Python 3.11
RUN conda install mamba -c conda-forge -y && \
    mamba create -n team2_env python=3.10 -y && \
    mamba install --yes --file requirements.txt && \
    mamba clean --all -f -y

# Set the environment path to use team2_env
ENV PATH="/root/miniconda3/envs/team2_env/bin:$PATH"

# Copy NGINX config and application files
COPY nginx.conf /etc/nginx/nginx.conf
COPY UI/ .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Expose necessary ports
EXPOSE 80
EXPOSE 5002
EXPOSE 6002

# Start NGINX, Streamlit, and Jupyter Notebook
CMD ["sh", "-c", "service nginx start && \
    streamlit run app.py --server.port=5002 & \
    jupyter notebook --ip=0.0.0.0 --port=6002 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' & \
    tail -f /dev/null"]
