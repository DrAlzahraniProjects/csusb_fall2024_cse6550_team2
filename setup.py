import os
import subprocess
import sys
from dotenv import set_key, load_dotenv

def run_command(command, error_message):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{error_message}: {e}")
        sys.exit(1)

def stop_and_remove_containers(ports):
    """Stop and remove Docker containers running on specific ports."""
    print("Stopping and removing Docker containers on ports:", ports)
    for port in ports:
        try:
            result = subprocess.check_output(f"docker ps -q -f publish={port}", shell=True).decode().strip()
            if result:
                print(f"Stopping container on port {port}...")
                run_command(f"docker stop {result}", f"Failed to stop container on port {port}")
                print(f"Removing container on port {port}...")
                run_command(f"docker rm {result}", f"Failed to remove container on port {port}")
            else:
                print(f"No container found running on port {port}.")
        except subprocess.CalledProcessError as e:
            print(f"Error while checking containers on port {port}: {e}")
            sys.exit(1)

def store_api_key_in_env(api_key):
    """Store the provided API key in the .env file."""
    env_path = os.path.join(os.getcwd(), ".env")

    # Create .env file if it does not exist
    if not os.path.exists(env_path):
        with open(env_path, "w") as env_file:
            env_file.write("")

    # Load existing .env file
    load_dotenv(dotenv_path=env_path)

    # Store the API key
    set_key(env_path, "API_KEY", api_key)
    print("API key stored in .env file successfully!")

def build_docker_image():
    """Build the Docker image."""
    print("Building the Docker image...")
    run_command("docker build -t team2-app .", "Failed to build Docker image")

def run_docker_container():
    """Run the Docker container."""
    print("Running the Docker container...")
    run_command("docker run -d -p 5002:5002 -p 6002:6002 -e API_KEY=${API_KEY} team2-app", "Failed to run Docker container")
    print("Docker container started successfully!")
    print("You can now access the application:")
    print("Website: http://localhost:5002/team2")

def main():
    print("Starting cross-platform automation script.....")

    # Step 1: Stop and remove existing Docker containers
    ports = [5002, 6002]
    stop_and_remove_containers(ports)

    # Step 2: Retrieve API key from environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API_KEY environment variable is not set.")
        sys.exit(1)

    # Step 3: Store API key in .env file
    store_api_key_in_env(api_key)

    # Step 4: Build the Docker image
    build_docker_image()

    # Step 5: Run the Docker container
    run_docker_container()

if __name__ == "__main__":
    main()
