# CSE Academic Advisor Chatbot (Team 2)
CSE 6550: Software Engineering Concepts, Fall 2024

**California State University, San Bernardino**

## Project Overview

The **CSE Academic Advisor Chatbot** is an AI-driven assistant designed to help students navigate academic queries related to the Computer Science and Engineering (CSE) department. The chatbot can answer questions about course prerequisites, academic schedules, department policies, and more. This project leverages advanced AI models and various technologies to provide accurate and helpful information.


## Setup

To set up the **CSE Academic Advisor Chatbot** on your local machine, follow the steps below:

### Step 1: Clone the Repository

First, clone the GitHub repository to your local machine using the command below:

```
git clone https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team2.git
```

### Step 2: Navigate to the Project Directory

Once the repository is cloned, navigate to the project directory:

```
cd csusb_fall2024_cse6550_team2
```

### Step 3: Update the Local Repository

Ensure your local repository is up-to-date by running:

```
git pull origin main
```

### Step 4: Build and Run the Docker Image

Make sure Docker is installed and running on your machine. Build the Docker image using the following command:

```
docker-compose up --build -d
```

### Step 5: Access the application

**Windows** - Make sure Docker Desktop application is open in your local machine
Development : http://localhost:5002/team2/

Production : https://sec.cse.csusb.edu/team2/

### Step 6: Only for Development 
Delete only the container not images if you make any changes made in app.py.
No Need to re build the image. Use the below command if you make any change in app.py.

```
streamlit run UI/app.py

```

While Debudding, inorder to view any print statements

```
docker-compose logs -f team2-app

```


