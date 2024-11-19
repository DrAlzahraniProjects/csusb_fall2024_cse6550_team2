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
docker run -p 5002:5002 -p 6002:6002 -e API_KEY=MISTRAL_API_KEY  team2_app
```
### Step 5: Access the Chatbot Application

**Development** : http://localhost:5002/team2/

**Production** : https://sec.cse.csusb.edu/team2/

**Jupyter** : http://localhost:6002/team2/jupyter

## SQA Table

| **Answerable**                                                     | **Unanswerable**                                                        |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| Information about CSE programs?         | What job positions do alumni hold at tech companies?           |
| What accreditation do computer science programs have?         | How much funding is allocated for research projects in the school?           |
| How does CSE engage with industry for internships and employment?         | What are the housing options available near the school?           |
| Who is the current Chair of CSE, and how can they be contacted?         | Explain the prerequisites for courses offered at other universities?           |
| How can students access remote labs?         | What are the grading percentages for each class in the Computer Science program?           |
| What is the CSE Club, and what resources does it provide to students?         | Can you give the syllabus for every course offered by CSE?           |
| What research and internship opportunities are available?         | What types of equipment can be checked out by students?           |
| How can students apply for the Excels Scholarship?         | Is parking free at the university?           |
| Where is the School of Computer Science and Engineering located?         | What school activities or events take place in the Winter?           |
| What are the contact details for the IT support staff?        | What are the most popular elective courses in the school?          |




