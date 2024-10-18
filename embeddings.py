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
import torch
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from data_scraping import cleaned_webpage_contents

api_key = "IetRnH5Lb578MdB5Ml0HNTdMBzeHUe7q"
model = "mistral-embed"
client = Mistral(api_key=api_key)
corrected_embeddings = []

def create_data_embeddings():
    # Generate embeddings from the cleaned content using Mistral API
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=cleaned_webpage_contents  # Pass cleaned webpage content for embedding
    )
    #print(type(cleaned_webpage_contents))
    # Extract embeddings from the API response
    embedding_list = [item.embedding for item in embeddings_batch_response.data]
    for idx, embedding in enumerate(embedding_list):
        if len(embedding) != 768:
            print(f"Embedding {idx} has incorrect length: {len(embedding)}")
        #print(embedding_list,"el")
        #print(f"Embedding {idx}: {embedding[:10]}")  # Print first 10 values for brevity
        #print(f"Type of embedding {idx}: {type(embedding[0])}")
    for embedding in embedding_list:
        if len(embedding) > 768:
            # Truncate the embedding to 768 dimensions
            corrected_embeddings.append(embedding[:768])
        elif len(embedding) < 768:
            # Pad the embedding with zeros to reach 768 dimensions
            corrected_embeddings.append(embedding + [0.0] * (768 - len(embedding)))
        else:
            # Already correct dimension
            corrected_embeddings.append(embedding)
    if len(cleaned_webpage_contents) != len(embedding_list):
        #print(f"Mismatch detected! Contents: {len(cleaned_webpage_contents)}, Embeddings: {len(embedding_list)}")

        # Find the minimum length
        min_length = min(len(cleaned_webpage_contents), len(embedding_list))

        # Truncate both lists to match the smaller length
        cleaned_webpage_contents = cleaned_webpage_contents[:min_length]
        embedding_list = embedding_list[:min_length]

        #print(f"Lengths after adjustment: Contents: {len(cleaned_webpage_contents)}, Embeddings: {len(embedding_list)}")
    else:
        print("No mismatch in lengths. Ready to insert data into Milvus.")