from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from web_crawler import scrape_main_page
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
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Constants and Parameters
nltk.download('punkt')

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
    """Retrieve relevant context from Milvus along with their source URLs."""
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text_content", "url"]
    )

    # Prepare text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chunk size to fit within token limits
        chunk_overlap=100  # Overlap for continuity between chunks
    )

    # Prepare context chunks with sources
    context_chunks = []
    for result in results[0]:  # Iterate through the top results
        # Access fields directly
        text_content = result.entity.get("text_content")  # Use .get() directly on the result
        url = result.entity.get("url")  # Use .get() directly on the result

        if not text_content or not url:
            continue

        # Split text content into manageable chunks
        text_splits = text_splitter.split_text(text_content)

        # Add each chunk with its associated URL
        for split in text_splits:
            context_chunks.append({"text_content": split, "url": url})

    # print("Retrieved Context Chunks:", context_chunks)
    return context_chunks


def extract_keywords(query, context):
    """Extract keywords dynamically using TF-IDF."""

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
    vectorizer.fit([query, context])
    return vectorizer.get_feature_names_out()

# def format_response_with_highlights(response, keywords, sources):
#     """Highlight keywords in the sources section only and add them to the response."""
#     # Highlight keywords in the sources
#     def highlight_keyword_in_source(source):
#         """Highlight keywords in a source link text."""
#         for keyword in keywords:
#             source = re.sub(
#                 f"\\b{re.escape(keyword)}\\b",
#                 f"<span style='background-color: yellow;'>{keyword}</span>",
#                 source,
#                 flags=re.IGNORECASE
#             )
#         return source

#     # Format sources with highlighted keywords
#     if sources:
#         highlighted_sources = [
#             f"<a href='{source}' target='_blank'>{highlight_keyword_in_source(source)}</a>"
#             for source in sources
#         ]
#         sources_html = ", ".join(highlighted_sources)
#         response += f"\n\n<b>Sources:</b> {sources_html}"

#     return response

# def format_response_with_highlights(response, keywords, sources):
#     """Highlight keywords in the response but ensure the entire source URL is clickable."""
#     # Highlight keywords in the display text only, not the actual clickable URL
#     def highlight_keyword_in_display(source):
#         """Highlight keywords in the display text."""
#         display_text = source  # Use the raw URL as the display text
#         for keyword in keywords:
#             display_text = re.sub(
#                 f"\\b{re.escape(keyword)}\\b",
#                 f"<span style='background-color: yellow;'>{keyword}</span>",
#                 display_text,
#                 flags=re.IGNORECASE
#             )
#         return display_text

#     # Add only the first source if available
#     if sources:
#         first_source = sources[0]
#         highlighted_display_text = highlight_keyword_in_display(first_source)  # Highlight keywords in the display text only
#         response += f"\n\n<b>Sources:</b> <a href='{first_source}' target='_blank'>{highlighted_display_text}</a>"

#     return response
def format_response_with_highlights(response, keywords, sources):
    """Add plain clickable sources to the response without highlighting."""
    # Add only the first source if available
    if sources:
        first_source = sources[0]  # Use only the first source
        response += f"\n\n<b>Sources:</b> <a href='{first_source}' target='_blank'>{first_source}</a>"

    return response


def invoke_llm_for_response(query):
    """Generate a response with highlighted keywords and exclude sources if no context is found."""
    llm = ChatMistralAI(model='open-mistral-7b', api_key=os.getenv("API_KEY"))

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the context: {context}\nAnswer the question: {question}"
    )
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate embedding for the query
    query_embedding = np.array(model.encode(query), dtype=np.float32).tolist()

    # Retrieve contexts with associated sources
    collection = Collection("CSUSB_CSE_Data")
    context_chunks = retrieve_context(query_embedding, collection)

    # Combine chunks while keeping within the token limit
    max_tokens = 3000  # Model's max input tokens (adjust for safety)
    current_tokens = 0
    selected_chunks = []
    for chunk in context_chunks:
        chunk_tokens = len(chunk["text_content"].split())  # Approximation of tokens
        if current_tokens + chunk_tokens > max_tokens:
            break
        selected_chunks.append(chunk["text_content"])
        current_tokens += chunk_tokens

    # Prepare context and deduplicate sources
    context = " ".join(selected_chunks)
    sources = list(set(chunk["url"] for chunk in context_chunks)) if selected_chunks else []  # Only include sources if context exists
    # print(selected_chunks,"selected_chunks...")
    # If no context is found, pass a generic fallback context to the LLM
    if not selected_chunks:
        fallback_context = (
            "The context does not provide specific information relevant to the query. "
            "Please generate a general response to help the user."
        )
        response = rag_chain.invoke({"context": fallback_context, "question": query})
        # Do not include sources in the response
        return format_response_with_highlights(response, [], [])

    # Generate the response using the retrieved context
    response = rag_chain.invoke({"context": context, "question": query})

    # Highlight keywords and format response
    keywords = extract_keywords(query, response)
    final_response = format_response_with_highlights(response, keywords, sources)

    return final_response
