# Academic Advisor Chatbot 

The **Academic Advisor Chatbot** project is an AI-driven chatbot designed to help students navigate academic queries related to the Computer Science and Engineering (CSE) department. The chatbot can answer questions about course prerequisites, academic schedules, department policies, and more.


## **Table of Contents**

To set up the project on your local machine, follow the steps below:

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Accessing the Application](#accessing-the-application)
* [Software Quality Assurance](#software-quality-assurance)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)

---
## Prerequisites

Make sure the following are installed on your local machine before proceeding:

- [Docker](https://www.docker.com/products/docker-desktop/)

---
## Installation

#### Step 1: Clone the Repository

First, clone the GitHub repository to your local machine using the command below:

```
git clone https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team2.git
```


#### Step 2: Navigate to the Project Directory

Once the repository is cloned, navigate to the project directory location:

```
cd csusb_fall2024_cse6550_team2
```

#### Step 3: Update the Local Repository

Ensure your local repository is up-to-date by running the below command:

```
git pull origin main
```

#### Step 4: Build and Run the Docker Image

Make sure Docker is installed and running on your local machine. Build the Docker image by using the following command:

```
docker build -t team2_app .
```

Run the Docker image by using the following command. Make sure you replace the "MISTRAL_API_KEY" with actual key from [Team2 Discussions Board](https://csusb.instructure.com/courses/43192/discussion_topics/419700) 
```
docker run -d -p 5002:5002 -p 6002:6002 -e API_KEY=MISTRAL_API_KEY  team2_app
```
---
## Accessing the Application

Once the docker container starts running, you can access the Academic Advising Chatbot at:

**Accessing Locally Through Docker** : 

*App* - http://localhost:5002/team2/

*Jupyter* -  http://localhost:6002/team2/jupyter

**Accessing Through CSE Web Server** : 

*App* - https://sec.cse.csusb.edu/team2/

*Jupyter* - https://sec.cse.csusb.edu/team2/jupyter

---

## Software Quality Assurance 

This section includes a set of questions that our chatbot can and cannot answer.

| **Answerable Questions**                                                                 | **Unanswerable Questions**                                                |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| What are the faculty office hours of Dr. Khalil Dajani?                                  | Tell me about the Sociology department?                                   |
| Give me the contact information for the CSE department.                                  | Do you know where the HR office is located?                              |
| What certificate programs are available?                                                | How can I use the weblogon service?                                      |
| Provide a list of faculty and staff in the CSE department.                               | What is the history of CSUSB?                                            |
| How can I make an advising appointment?                                                 | Tell me about the Entrepreneurship programs.                             |
| How can I check my class schedule?                                                      | Can you name the public administration faculty members?                  |
| What courses are offered for undergraduate students?                                     | What are the Art and Letter perspectives?                       |
| What are the student internship opportunities?                                           | Is IST a course in the CSE department?                                   |
| How do I get remote lab access?                                                         | What is ATI?                                                             |
| How can I join the CSE club?                                                            | Where are the CSU FILA awards held?                                      |
| Can I install VMware Horizon Client?                                                    |                                                                           |

---

## Troubleshooting
- If you encounter issues while building or running the container, ensure that Docker is installed and running correctly.
- Make sure you have ample space in your local machine to avoid issues while building docker image.
- Ensure the port 5002 is not being used by another application.

---

## Contributors
- CSUSB Fall 2024 CSE6550 Team-2
