import os
import subprocess
import sys

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
            # Find the container ID using the port
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

def build_docker_image():
    """Build the Docker image."""
    print("Building the Docker image...")
    run_command("docker build -t team2_app .", "Failed to build Docker image")



def run_docker_container(mistral_key):
    """Run the Docker container."""
    print("Running the Docker container...")
    run_command(f"docker run -d -p 5002:5002 -p 6002:6002 -e API_KEY={mistral_key} team2_app", "Failed to run Docker container")
    print("Docker container started successfully!")
    print("You can now access the application:")
    print("Website: http://localhost:5002/team2")
    print("Wait 30 seconds more when accessing the webserver.")  # Additional message

def main():
    print("Starting cross-platform automation script.....")

    # Step 1: Stop and remove existing Docker containers
    ports = [5002]
    stop_and_remove_containers(ports)

    # Step 2: Build the Docker image
    build_docker_image()

    # Step 3: Run the Docker container with Mistral API key
    mistral_key = input("Enter your Mistral API key: ")
    run_docker_container(mistral_key)

if __name__ == "__main__":
    main()
