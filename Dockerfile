# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Copy the assets folder (including style.css)
COPY assets/ /app/assets/

# Install wget and other necessary tools
RUN apt-get update && apt-get install -y wget

# Determine system architecture and install the corresponding version of Miniconda
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
    echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash Miniconda3-latest-Linux-*.sh -b && \
    rm Miniconda3-latest-Linux-*.sh && \
    apt-get clean

# Install Mamba using Miniconda and create a new environment with Python 3.11
RUN /root/miniconda3/bin/conda install mamba -c conda-forge -y && \
    /root/miniconda3/bin/mamba create -n team2_env python=3.9 -y && \
    /root/miniconda3/bin/mamba clean --all -f -y

# Set environment path to use team2_env and ensure bash is used
ENV PATH="/root/miniconda3/envs/team2_env/bin:$PATH"

# Check if the environment exists and is activated, and then install dependencies
RUN /root/miniconda3/bin/conda info --envs && \
    echo "source /root/miniconda3/bin/activate team2_env" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && mamba install --yes --file /app/requirements.txt && mamba clean --all -f -y"

# Install sentence-transformers using pip
# RUN /bin/bash -c "source ~/.bashrc && pip install sentence-transformers"

# Install Jupyter Notebook
RUN /bin/bash -c "source ~/.bashrc && mamba install -c conda-forge jupyter"

# Install pymilvus using Mamba
# RUN /bin/bash -c "source ~/.bashrc && mamba install -c conda-forge pymilvus && mamba clean --all -f -y"
RUN /bin/bash -c "source ~/.bashrc && pip install pymilvus"

# Check if pymilvus is installed 
RUN /bin/bash -c "source ~/.bashrc && python -c 'import pymilvus'"

# Install sentence-transformers using pip
RUN /bin/bash -c "source ~/.bashrc && pip install sentence-transformers"

# Copy the app code from the UI folder
COPY UI/ .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Expose ports for Streamlit
EXPOSE 5002

# Run the Streamlit app
# CMD ["sh", "-c", "streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2"]
# 
CMD ["/bin/bash", "-c", "source /root/miniconda3/bin/activate team2_env && streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2"]