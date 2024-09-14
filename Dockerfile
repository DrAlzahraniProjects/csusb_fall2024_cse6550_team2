# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install Miniconda
RUN apt-get update && apt-get install -y wget \
    && if [ "$(uname -m)" = "aarch64" ]; then \
           wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh; \
       else \
           wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
       fi \
    && bash miniconda.sh -b \
    && rm miniconda.sh \
    && apt-get clean   

# Install Mamba using Miniconda
RUN /root/miniconda3/bin/conda install mamba -c conda-forge

# Create a new environment with Python 3.11
RUN /root/miniconda3/bin/mamba create -n team2_env python=3.11 -y

# Set environment path
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

# Copy the current directory contents into the container
COPY . /app

# Expose ports for Streamlit and Jupyter
EXPOSE 5002
EXPOSE 8888

# Start both Streamlit and Jupyter
CMD ["sh", "-c", "streamlit run app.py --server.port=5002 & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
