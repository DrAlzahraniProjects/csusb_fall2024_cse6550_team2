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
        score = result.score  # Similarity score from Milvus

        if not text_content or not url:
            continue

        # Split text content into manageable chunks
        text_splits = text_splitter.split_text(text_content)

        # Add each chunk with its associated URL
        for split in text_splits:
            context_chunks.append({"text_content": split, "url": url,"score": score})

    context_chunks = sorted(context_chunks, key=lambda x: x["score"], reverse=True)
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
    PROMPT_TEMPLATE = """
    Based on the context: {context}\nAnswer the question: {question}. 
    If you cannot answer the question, please just say: 
    "I don't have enough information to answer this question."
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
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

    # Combine the most relevant chunk (first in sorted order) for the context
    if context_chunks:
        most_relevant_chunk = context_chunks[0]  # Select the highest-scoring chunk
        context = most_relevant_chunk["text_content"]
        sources = [most_relevant_chunk["url"]]  # Use the URL of the most relevant chunk
    else:
        context = ""
        sources = []

    # If no context is found, pass a generic fallback context to the LLM
    if not context:
        fallback_context = (
            "I don't have enough information to answer this question."
        )
        response = rag_chain.invoke({"context": fallback_context, "question": query})
        return format_response_with_highlights(response, [], [])

    # Generate the response using the retrieved context
    response = rag_chain.invoke({"context": context, "question": query})

    # Define fallback phrases to exclude sources
    fallback_phrases = [
        "I don't have enough information to answer this question.",
        "The context does not provide information",
        "The context does not provide specific information",
        "you may want to visit the university's official website",
    ]

    # Check if response contains any fallback phrases
    if any(phrase in response for phrase in fallback_phrases) or not response.strip():
        # Ensure the response does not include sources
        return format_response_with_highlights(response, [], [])

    # Highlight keywords and format the response with sources if relevant
    keywords = extract_keywords(query, response)
    final_response = format_response_with_highlights(response, keywords, sources)
    return final_response

    