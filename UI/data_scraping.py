# from pymilvus import MilvusClient, model, connections, db
import requests
import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from mistralai import Mistral
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Milvus
# from langchain.embeddings import MistralEmbeddings
# from langchain.llms import Mistral
from langchain.chains import RetrievalQA
from pymilvus import connections, Collection
import numpy as np
# from langchain_community.llms import Mistral
import streamlit as st
#import torch
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Webpage URLs list
webpage_urls = [
    "https://www.csusb.edu/cse/programs"
    "https://www.csusb.edu/cse",
    "https://www.csusb.edu/cse/programs/bs-computer-science",
    "https://www.csusb.edu/cse/programs",
    "https://www.csusb.edu/cse/programs/bs-computer-engineering",
    #"https://www.csusb.edu/cse/programs/bs-bioinformatics",
    #"https://www.csusb.edu/cse/programs/ba-computer-systems",
    #"https://www.csusb.edu/cse/programs/ms-computer-science",
    #"https://www.csusb.edu/cse/programs/minor-computer-science",
    #"https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/data-sci-minor/",
    #"https://www.csusb.edu/profile/alzahran",
    #"https://www.csusb.edu/cse/faculty-and-staff/faculty-office-hours",
    #"https://www.csusb.edu/cse/faculty-and-staff/industry-advisory-board",
    #"https://catalog.csusb.edu/coursesaz/cse/",
    #"https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/computer-science-ms/",
    "https://www.csusb.edu/cse/faculty-staff",
    #"https://www.csusb.edu/cse/advising",
    #"https://www.csusb.edu/cse/resources",
    #"https://www.csusb.edu/cse/internships-careers",
    #"https://www.csusb.edu/cse/contact"
]

# Function to fetch and extract text from a webpage
def get_webpage_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Clean and extract the main textual content from the page
            text = clean_webpage_text(str(soup))
            return text
        else:
            #print(f"Failed to fetch {url}: {response.status_code}")
            return None
    except Exception as e:
        #print(f"Error fetching {url}: {str(e)}")
        return None

# Function to clean and process webpage text
def clean_webpage_text(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')

    # Remove unnecessary tags
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Extract cleaned text
    text = soup.get_text(separator=' ', strip=True)

    # Remove multiple spaces and limit to 5000 characters
    text = re.sub(r'\s+', ' ', text)
    return text[:5000]

# Fetch and clean the webpage content
cleaned_webpage_contents = []
for url in webpage_urls:
    content = get_webpage_text(url)
    if content:
        cleaned_webpage_contents.append(content)
    else:
        cleaned_webpage_contents.append("")

