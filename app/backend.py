from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from WebCrawler import scrape_main_page
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_mistralai.chat_models import ChatMistralAI
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
    
)
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
import nltk
import os
from urllib.parse import urljoin,urlparse
from scipy.sparse import csr_matrix
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models import QAModel
import torch

# Constants and Parameters
nltk.download('punkt')
# Switch between models to get optimized information retrieval on QA tasks
# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_api_key():
    """Retrieve the API key from the environment."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Ensure the API key is set in main.py before proceeding.")
    return api_key

# Function to retrieve and split context from Milvus
def retrieve_context(query_embedding, collection: Collection, limit=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text_content"]
    )

    # Combine retrieved documents into a single context string
    combined_context = " ".join([res.text_content for result in results for res in result])
    # Text splitter configuration to chunk context text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Set chunk size based on LLM's token limit
        chunk_overlap=100  # Set overlap to ensure continuity between chunks
    )
    # Split combined context into manageable chunks
    context_chunks = text_splitter.split_text(combined_context)
    return context_chunks

# Function to invoke the language model for generating a response
def invoke_llm_for_response(query: str):
    api_key = get_api_key()
    if not isinstance(query, str):
        raise ValueError("The input query must be a string.")
    
    # Initialize the language model
    llm = ChatMistralAI(model='open-mistral-7b', api_key=api_key)
    
    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the context: {context}\nAnswer the question: {question}"
    )

    # Define the RAG Chain
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Convert the query to embedding
    #query_embedding = np.array(model.encode(query), dtype=np.float32).tolist()  # Ensure float32 format
    query_embedding = model.encode(query)

    # Initialize NeMo Curator model
    nemo_curator = QAModel.from_pretrained(model_name="qa_squadv1.1_bertbase")

    # Retrieve context from Milvus and select chunks within token limits
    collection = Collection("CSUSB_CSE_Data")  # Initialize your collection
    formatted_content_chunks = retrieve_context(query_embedding, collection)
    context_within_limit = " ".join(formatted_content_chunks[:3])  # Limit context to the first few chunks if needed


   # Convert the context to embedding
    context_embeddings = model.encode(context_within_limit)

    # Concatenate query and context embeddings
    input_ids = np.concatenate([query_embedding, context_embeddings])

    # Create attention mask (1 for all tokens)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    # Create token type ids (0 for query tokens, 1 for context tokens)
    token_type_ids = np.concatenate([np.zeros_like(query_embedding, dtype=np.int64), np.ones_like(context_embeddings, dtype=np.int64)])

    # Convert to torch tensors and add batch dimension
    input_ids = torch.tensor([input_ids], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).unsqueeze(0)

    # Initialize NeMo Curator model with an available model name
    nemo_curator = QAModel.from_pretrained(model_name="qa_squadv1.1_bertbase")

    # Prepare the input for the model
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }
    # Generate response using NeMo Curator
    nemo_response = nemo_curator(**inputs)

    # Process the response
    response = nemo_response[0]['answer']

    # Print and return the final response
    print("Final Response:", response)
    return response