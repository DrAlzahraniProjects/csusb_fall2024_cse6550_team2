# from pymilvus import MilvusClient, model, connections, db
import requests
import os
from milvus_utils import create_collection
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
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Load the Mistral model and tokenizer from Hugging Face
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_llm = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model_llm, tokenizer=tokenizer)
# Set up Mistral API
api_key = "IetRnH5Lb578MdB5Ml0HNTdMBzeHUe7q"
model = "mistral-embed"
client = Mistral(api_key=api_key)

def generate_prompt(context,input):
    """
    Create a prompt template for the RAG model

    Returns:
        PromptTemplate: The prompt template for the RAG model
    """
    # Define the prompt template
    PROMPT_TEMPLATE = """
    Human: You are an AI academic assistant, specialized in providing detailed and accurate answers.
    Your responses should be well-researched, supported by statistical data, and formatted in a clear and concise manner.
    Use only the information provided within the <context> tags to answer the question enclosed in <question> tags.
    Ensure that the answer is directly derived from the given context and cite relevant details when possible.
    If the information is not present in the context, respond with "I don't know" and do not attempt to fabricate an answer.

    Guidelines:
    - Provide a direct answer to the question using specific data or references from the context.
    - Where possible, include numerical data, statistics, or dates to support your response.
    - Be clear and concise, avoiding unnecessary elaboration.
    - Maintain a formal and academic tone.

    <context>
        {context}
    </context>

    <question>
        {input}
    </question>

    Assistant:"""

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    print("Prompt Created")

    return prompt

def adjust_query_embedding(query_embedding, target_dim=768):
    current_dim = len(query_embedding)

    if current_dim > target_dim:
        # Truncate the embedding if it's larger than the target dimension
        adjusted_embedding = query_embedding[:target_dim]
    elif current_dim < target_dim:
        # Pad the embedding with zeros if it's smaller than the target dimension
        adjusted_embedding = query_embedding + [0.0] * (target_dim - current_dim)
    else:
        # No adjustment needed
        adjusted_embedding = query_embedding

    return adjusted_embedding

def query_embeddings():
    # Generate query embedding using the Mistral client
    query_embedding = client.embeddings.create(model=model_name, inputs=[query]).data[0].embedding

    # Adjust the query embedding to match the target dimension of the collection (e.g., 768)
    adjusted_query_embedding = adjust_query_embedding(query_embedding, target_dim=768)
    # Convert to numpy array and reshape for search
    adjusted_query_embedding = np.array(adjusted_query_embedding).astype(np.float32).reshape(1, -1)

#  Function to search Milvus collection and generate a chain-based response
def generate_response(query, collection, client, model_name=model, generator=None):
    """
    Uses Milvus to retrieve relevant context and generates a response using an LLM.
    
    Args:
        query (str): The user's question.
        collection (Collection): The Milvus collection instance.
        client: The Mistral client instance for embedding generation.
        model_name (str): The model used for generating query embeddings.
        generator: The LLM instance for generating responses.
        f"Using the following context, provide a detailed answer to the user's question:\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n\n"
        f"Answer in a comprehensive manner, providing detailed information based on the context."
    
    Returns:
        str: The generated answer based on the chained context.
    """
    # Generate query embedding using the Mistral client
    query_embedding = client.embeddings.create(model=model_name, inputs=[query]).data[0].embedding

    # Adjust the query embedding to match the target dimension of the collection (e.g., 768)
    adjusted_query_embedding = adjust_query_embedding(query_embedding, target_dim=768)
    # Convert to numpy array and reshape for search
    adjusted_query_embedding = np.array(adjusted_query_embedding).astype(np.float32).reshape(1, -1)

    # Define search parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    # Search in Milvus collection
    results = collection.search(
        data=adjusted_query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["content"]
    )

    # Extract matched content from the search results
    matched_contents = []
    for hit in results:
        try:
            entity_content = hit.entity.get("content")
            matched_contents.append(entity_content)
        except AttributeError:
            continue

    # Combine matched content into a single context if available
    combined_context = "\n\n".join(matched_contents) if matched_contents else "No relevant context found."

    # Create a prompt using the matched content and user's input
    prompt = generate_prompt(combined_context,query)
    # Generate the response using the provided LLM generator
    response = generator(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
    print(response)
    return response

