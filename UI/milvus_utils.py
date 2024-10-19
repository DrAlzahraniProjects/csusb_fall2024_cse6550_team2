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
from UI.data_scraping import cleaned_webpage_contents
from UI.embeddings import corrected_embeddings, create_data_embeddings

@st.cache_resource
# Define the collection schema
def create_collection():
    collection_name = "academic_webpages"
    if utility.has_collection(collection_name):
        # If collection exists, load it
        collection = Collection(collection_name)
        #print(f"Collection '{collection_name}' loaded.")
    else:
        # Define fields for a new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust dimension based on Mistral
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000)  # Storing webpage content
        ]
        schema = CollectionSchema(fields, description="Collection for webpage embeddings")
        # Create a new collection
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection created: {collection.name}")
    return collection  # Return the collection object

def initialize_milvus():
    #  Establish connection
    connections.connect(host='localhost', port='19530')
    #  Create Collection
    collection = create_collection()
    # Create Index
    index_params = {
                "index_type": "IVF_FLAT",  # Index type
                "metric_type": "L2",       # Distance metric, can also be IP (Inner Product) for cosine similarity
                "params": {"nlist": 128}   # Number of clusters, adjust based on data
            }
    collection.create_index(field_name="embedding", index_params=index_params)
    try:
        if len(cleaned_webpage_contents) == len(corrected_embeddings):
            list_embeddings= create_data_embeddings()
            # Insert data into Milvus
            data = [
                # [i for i in range(len(content_list))],  # Auto-generated IDs (primary keys)
                corrected_embeddings,  # List of embeddings as FLOAT_VECTORs
                cleaned_webpage_contents  # List of webpage contents (VARCHARs)
            ]
            insert_result = collection.insert(data)
            #print(f"Inserted {len(content_list)} entries into Milvus.")

            # Flush to persist data
            collection.flush()
        else:
            print("Error: Content list and embedding list lengths do not match!")
    except Exception as e:
        print(f"Failed to insert data into Milvus: {e}")


def insert_vectors(cleaned_webpage_contents, corrected_embeddings):
    # Insert the cleaned webpage content and their corresponding embeddings into Milvus
    # Ensure the collection is created and loaded
   
    collection =create_collection()
    collection.load()

    try:
        if len(cleaned_webpage_contents) == len(corrected_embeddings):
            list_embeddings= create_data_embeddings()
            # Insert data into Milvus
            data = [
                # [i for i in range(len(content_list))],  # Auto-generated IDs (primary keys)
                corrected_embeddings,  # List of embeddings as FLOAT_VECTORs
                cleaned_webpage_contents  # List of webpage contents (VARCHARs)
            ]
            insert_result = collection.insert(data)
            #print(f"Inserted {len(content_list)} entries into Milvus.")

            # Flush to persist data
            collection.flush()
        else:
            print("Error: Content list and embedding list lengths do not match!")
    except Exception as e:
        print(f"Failed to insert data into Milvus: {e}")

