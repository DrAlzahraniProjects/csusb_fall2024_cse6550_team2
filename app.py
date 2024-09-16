import streamlit as st
import os
import subprocess

def main():
    st.title("Hello World! - Team 2")

if __name__ == "__main__": #needs to relode on windows 11 
    # Check if streamlit instance is running
    if os.environ.get("STREAMLIT_RUNNING") == "1":
        main() 
    else:
        # Set the environment variable to indicate Streamlit is running
        os.environ["STREAMLIT_RUNNING"] = "1" 
        subprocess.run(["streamlit", "run", __file__, "--server.port=5002", "--server.address=0.0.0.0"])
