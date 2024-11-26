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

Make sure Docker is installed and running on your machine. Build and run the Docker image using the following command:

```
docker build -t team2_app .
```
Note: Use Mistral API Key in place of MISTRAL_API_KEY here below

```
docker run -d -p 5002:5002 -p 6002:6002 -e API_KEY=MISTRAL_API_KEY  team2_app
```
### Step 5: Access the Chatbot Application

**Development** : 
*App* - http://localhost:5002/team2/
*Jupyter* -  http://localhost:6002/team2/jupyter

**Production** : 
*App* - https://sec.cse.csusb.edu/team2/
*Jupyter* - https://sec.cse.csusb.edu/team2/jupyter


## SQA Table

| **Answerable Questions**                                                                 | **Unanswerable Questions**                                                |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
|                                   | Tell me about the Sociology department?                                   |
| Give me the contact information for the CSE department.                                  | Do you know where the HR office is located?                              |
| What certificate programs are available?                                                | How can I use the weblogon service?                                      |
| Provide a list of faculty and staff in the CSE department.                               | What is the history of CSUSB?                                            |
| How can I make an advising appointment?                                                 | Tell me about the Entrepreneurship programs.                             |
| How can I check my class schedule?                                                      | Can you name the public administration faculty members?                  |
| What courses are offered for undergraduate students?                                     | What are the Art and Letter perspectives at CSUSB?                       |
| What are the student internship opportunities?                                           | Is IST a course in the CSE department?                                   |
| How do I get remote lab access?                                                         | What is ATI?                                                             |
| How can I join the CSE club?                                                            | Where are the CSU FILA awards held?                                      |
| Can I install VMware Horizon Client?                                                    |                                                                           |




