from pymilvus import Collection
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
import nltk
import os
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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
        text_content = result.entity.get("text_content")
        url = result.entity.get("url")

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

def format_response_with_highlights(response, keywords, sources):
    # print("Response:", response)
    """Add a single clickable source to the response."""
    if sources and sources[0]:  # Check if a single source is available
        response += f"\n\n<b>Sources:</b> <a href='{sources[0]}' target='_blank'>{sources[0]}</a>"

    return response

# Function to invoke the language model for generating a response
def invoke_llm_for_response(query):
    """Generate a response with highlighted keywords and exclude sources if no information is provided."""
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
    sources = list({chunk["url"] for chunk in context_chunks}) if selected_chunks else []

    # If no context is found, pass a generic fallback context to the LLM
    if not selected_chunks:
        fallback_context = (
            "The context does not provide specific information relevant to the query. "
            "Generate a general response to help the user."
        )
        response = rag_chain.invoke({"context": fallback_context, "question": query})
        # Do not include sources in the response
        return format_response_with_highlights(response, [], [])

    # Generate the response using the retrieved context
    response = rag_chain.invoke({"context": context, "question": query})

    # Define fallback phrases to exclude sources
    fallback_phrases = [
        "The context does not provide information",
        "The context does not provide specific information",
        "you may want to visit the university's official website"
    ]

    # Check if response contains any fallback phrases
    if any(phrase in response for phrase in fallback_phrases) or not response.strip():
        # Return the response without sources
        final_response =format_response_with_highlights(response, [], [])
        return final_response
    else:
        # Highlight keywords and format response
        keywords = extract_keywords(query, response)
        final_response = format_response_with_highlights(response, keywords, sources)
        return final_response


