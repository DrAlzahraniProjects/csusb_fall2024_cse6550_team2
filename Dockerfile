# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the assets folder (including style.css)
COPY assets/ /app/assets/

# Install dependencies
RUN apt-get update && apt-get install -y wget


# Install Langchain-milvus
RUN pip install -U langchain-milvus

RUN pip install langchain
#
RUN pip install --upgrade --quiet  langchain langchain-core langchain-community langchain-text-splitters langchain-milvus langchain-openai bs4

# Determine system architecture and we will install the corresponding version of Miniconda
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash Miniconda3-latest-Linux-*.sh -b && \
    ls -la /root/miniconda3 && \
    rm Miniconda3-latest-Linux-*.sh && \
    apt-get clean

# Install Mamba using Miniconda and create a new environment with Python 3.11
RUN /root/miniconda3/bin/conda install mamba -c conda-forge -y \
    && /root/miniconda3/bin/mamba create -n team2_env python=3.11 -y \
    && /root/miniconda3/bin/mamba clean --all -f -y

# Set environment path to use team2_env and ensure bash is used
ENV PATH="/root/miniconda3/envs/team2_env/bin:$PATH"

# Activate the environment and install packages from requirements.txt
SHELL ["/bin/bash", "-c"]
RUN echo "source /root/miniconda3/bin/activate team2_env" >> ~/.bashrc

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN /bin/bash -c "source ~/.bashrc && mamba install --yes --file /app/requirements.txt && mamba clean --all -f -y"

# Install Jupyter Notebook
RUN /bin/bash -c "source ~/.bashrc && mamba install -c conda-forge jupyter"

# Install NGINX
# RUN apt-get update && apt-get install -y nginx

# Copy NGINX config
# COPY nginx.conf /etc/nginx/nginx.conf

# Copy the current directory contents into the container
COPY UI/ . 

# Setting environment variables for StreamLit
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Expose ports for NGINX, Streamlit, and Jupyter
# EXPOSE 82 
EXPOSE 5002
# EXPOSE 6002
# Start NGINX, Streamlit, and Jupyter using JSON array syntax for CMD
# Jupyter runs with token authentication disabled on port 6002
# ENTRYPOINT [ "python"]
# CMD ["streamlit" "run" "app.py"]

CMD ["sh", "-c", "streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2", "tail -f /dev/null"]