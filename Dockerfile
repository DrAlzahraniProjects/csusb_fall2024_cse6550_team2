# Use Python as the base image
FROM python:3.10-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN pip install torch

# Set the working directory in the container
WORKDIR /app

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
	wget \
	bzip2 \
	ca-certificates \
	build-essential \
    	python3-dev \
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

#API_KEY setup
ENV OPENAI_API_KEY=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

# Create a new environment with Python 3.10
RUN mamba create -n team2_env python=3.10 -y

# Activate the new environment
SHELL ["mamba", "run", "-n", "team2_env", "/bin/bash", "-c"]

RUN pip install -qU cython

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN mamba install --yes --file requirements.txt && mamba clean --all -f -y

RUN pip install nemo_toolkit[all]

# Install NeMo Guardrails
RUN pip install nemoguardrails

# RUN pip install -qU langchain_milvus langchain-cohere nemo-curator nemoguardrails
RUN pip install pymilvus[model] langchain langchain_community langchain_huggingface langchain_milvus beautifulsoup4 requests nltk langchain_mistralai sentence-transformers scipy

# Upgrade huggingface_hub to a specific version that includes ModelFilter
#RUN pip install huggingface-hub==0.24.0

# Install specific versions of related libraries
RUN pip install huggingface-hub==0.23.2 transformers==4.40.0

VOLUME /app/data
# Copy the current directory contents into the container at /app
COPY . /app

# Set the StreamLit ENV for configuration
ENV STREAMLIT_SERVER_BASEURLPATH=/team2
ENV STREAMLIT_SERVER_PORT=5002

# Streamlit port
EXPOSE 5002
# Jupyter Notebook port
EXPOSE 6002

# Add the conda environment's bin directory to PATH
ENV PATH=/opt/miniforge/envs/team2_env/bin:$PATH

# Run the Streamlit app and jupyter

CMD ["sh", "-c", "streamlit run app/main.py --server.port=5002 --server.address=0.0.0.0 --server.baseUrlPath=/team2 & jupyter notebook --ip=0.0.0.0 --port=6002 --no-browser --allow-root --NotebookApp.base_url=/team2/jupyter --NotebookApp.token=''"]

