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
import httpx

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
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text_content", "url"]
    )

    # Prepare text splitter
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,  # Chunk size to fit within token limits
    #     chunk_overlap=100  # Overlap for continuity between chunks
    # )
 
    # Prepare context chunks with sources
    context_chunks = []
    for result in results[0]:  # Iterate through the top results
        # Access fields directly
        text_content = result.entity.get("text_content")
        url = result.entity.get("url")
        similarity_score = 1 / (1 + result.distance)

        if not text_content or not url:
            continue

        # Split text content into manageable chunks
        # text_splits = text_splitter.split_text(text_content)

        # Add each chunk with its associated URL
        for split in text_content:
            context_chunks.append({"text_content": split, "url": url,"similarity_score": similarity_score})

    # print("Retrieved Context Chunks:", context_chunks)
    return context_chunks

def extract_keywords(query, context):
    """Extract keywords dynamically using TF-IDF."""

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
    vectorizer.fit([query, context])
    return vectorizer.get_feature_names_out()


def format_response_with_highlights(response, keywords, sources):
    """Add sources to the response only if they are valid."""
    if not sources or not response.strip():
        # If no sources or response is empty, return response without sources
        return response

    # Add the first valid source to the response
    response += f"\n\n<b>Sources:</b> <a href='{sources[0]}' target='_blank'>{sources[0]}</a>"
    return response
def invoke_llm_for_response(query):
    try:
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

        # Combine the most relevant chunk for the context
        if context_chunks:
            most_relevant_chunk = context_chunks[0]
            context = most_relevant_chunk["text_content"]
            sources = [most_relevant_chunk["url"]]
        else:
            context = "I don't have enough information to answer this question."
            sources = []  # Ensure no sources are attached

        # Generate the response using the retrieved context
        response = rag_chain.invoke({"context": context, "question": query})

        # Omit sources for fallback responses
        if not context or "I don't have enough information" in response:
            return format_response_with_highlights(response, [], [])

        # Highlight keywords and format the response with sources if relevant
        keywords = extract_keywords(query, response)
        return format_response_with_highlights(response, keywords, sources)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment before trying again."
        else:
            raise e
