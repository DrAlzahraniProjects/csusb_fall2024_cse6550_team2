# Use Mamba base image, which is compatible for Linux in Docker
FROM mambaorg/micromamba:1.4.2

# Set the working directory in the container
WORKDIR /app

# Copy environment.yml into the container
COPY environment.yml .

# Install dependencies via Mamba
RUN micromamba create --file environment.yml --name myenv --yes \
    && micromamba clean --all --yes

# Set the shell to use micromamba environment
SHELL ["micromamba", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy the entire project into the container
COPY . .

# Expose port 5002
EXPOSE 5002

# Run the application (Python file)
CMD ["micromamba", "run", "-n", "myenv", "streamlit", "run", "app.py", "--server.port=5002"]

