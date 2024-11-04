from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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


# Constants and Parameters
corpus_source = ["https://www.csusb.edu/cse","https://catalog.csusb.edu/"]
nltk.download('punkt')
MILVUS_URI = "./milvus_lite/milvus_vector.db"
# Switch between models to get optimized information retrieval on QA tasks
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_NAME_2 = "sentence-transformers/msmarco-distilbert-base-v3"
collection_name = "Academic_Webpages"
output_folder = "csusb_cse_content"

# Ensure directories exist
os.makedirs("milvus_lite", exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

def get_api_key():
    """Retrieve the API key from the environment."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Ensure the API key is set in main.py before proceeding.")
    return api_key

# Directory to store unreachable URLs
unreachable_dir = "unreachable_urls"
os.makedirs(unreachable_dir, exist_ok=True)

# Save unreachable URL to file
def save_unreachable_url(url):
    with open(os.path.join(unreachable_dir, "unreachable_urls.txt"), "a") as f:
        f.write(url + "\n")
   
# Function to load webpages and extract content
def load_webpages(url):
    try:
        response = requests.get(url, timeout=10)  # Setting a timeout
        response.raise_for_status()  # Check if the request was successful
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # print(f"Failed to connect to {url}")
        save_unreachable_url(url)
        return {"content": "", "source": url}
    except requests.exceptions.HTTPError as err:
        # print(f"HTTP error for {url}: {err}")
        save_unreachable_url(url)
        return {"content": "", "source": url}

    soup = BeautifulSoup(response.text, 'html.parser')
    content_list = []

    # Process <li> items and follow links to extract linked page content
    li_items = soup.find_all('li')
    for li in li_items:
        li_text = li.get_text()
        links = [a['href'] for a in li.find_all('a', href=True)]

        # Fetch content from each link in the <li>
        for link in links:
            linked_url = urljoin(url, link)
            linked_content_data = load_linked_content(linked_url)
            if linked_content_data:
                content_list.append(f"{li_text}: {linked_content_data}")

    # Extract text and links from paragraphs in the main page
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        paragraph_text = p.get_text()
        links = [a['href'] for a in p.find_all('a', href=True)]
        combined_text = f"{paragraph_text} (Links: {', '.join(links)})" if links else paragraph_text
        content_list.append(combined_text)

    # Combine content into a single text
    content = " ".join(content_list)
    return {"content": content, "source": url}

# Function to load content from linked pages
def load_linked_content(link_url):
    try:
        response = requests.get(link_url, timeout=10)  # Setting a timeout
        response.raise_for_status()
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # print(f"Failed to connect to {link_url}")
        save_unreachable_url(link_url)
        return ""
    except requests.exceptions.HTTPError as err:
        # print(f"HTTP error for {link_url}: {err}")
        save_unreachable_url(link_url)
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')
    content_list = []

    # Extract paragraphs and links from the linked page
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        paragraph_text = p.get_text()
        links = [a['href'] for a in p.find_all('a', href=True)]
        combined_text = f"{paragraph_text} (Links: {', '.join(links)})" if links else paragraph_text
        content_list.append(combined_text)

    # Combine the extracted content from the linked page
    content = " ".join(content_list)
    return content

# Split text into chunks for further processing
def split_text(text, chunk_size=20000):
    text_splitter = CharacterTextSplitter(separator=",", chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)

# Get texts data from the URLs
def get_texts_data(corpus_source):
    texts = []
    for url in corpus_source:
        page_data = load_webpages(url)
        content = page_data["content"]
        source = page_data["source"]

        if content:
            cleaned_content = content.replace("\n", " ")
            split_contents = split_text(cleaned_content)

            # Create a document entry with each chunk and the source
            for split_content in split_contents:
                texts.append({
                    "page_content": split_content,
                    "source": source
                })
    return texts

# Extract only the page content from the texts data
def extract_text_content(texts):
    return [text["page_content"] for text in texts if "page_content" in text]


texts = get_texts_data(corpus_source)
# Extract the cleaned text content from the dictionaries for embedding
text_contents = extract_text_content(texts)

# Initialize the dense and sparse embeddings
dense_embedding_func = HuggingFaceEmbeddings(model_name=MODEL_NAME_2)
dense_dim = len(dense_embedding_func.embed_query(text_contents[1]))
# print(dense_dim)
sparse_embedding_func = BM25SparseEmbedding(corpus=text_contents)
sparse_embedding_func.embed_query(text_contents[1])

# Initialize Milvus and create a collection
def initialize_milvus():
    global collection
    # Connect to Milvus
    connections.connect("default", uri=MILVUS_URI)
    print(f"Connected to Milvus at {MILVUS_URI}")

    # Check if the collection exists
    if utility.has_collection(collection_name):
        # Load the existing collection
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
        return collection
  
    else:
        print(f"Collection '{collection_name}' does not exist. Creating new collection.")

        # Define schema fields
        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        source_field = "source"

        # Define fields for the collection schema
        fields = [
            FieldSchema(
                name=pk_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=source_field, dtype=DataType.VARCHAR, max_length=500) 
        ]

        # Create the schema and the collection
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")
        print(f"Created collection '{collection_name}'.")

        # Create indexes for dense and sparse vectors
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}

        collection.create_index(dense_field, dense_index)
        collection.create_index(sparse_field, sparse_index)

        print(f"Created sparse vector index on '{dense_field} {sparse_field}'.")

        # Flush to persist changes
        collection.flush()
        print(f"Flushed collection '{collection_name}' to persist changes.")

    # Insert vectors into the collection
    entities = []
    
    for text in texts:
        text_content= str(text["page_content"])
        source = text["source"]
        entity = {
            "dense_vector": dense_embedding_func.embed_documents([text_content])[0],
            "sparse_vector": sparse_embedding_func.embed_documents([text_content])[0],
            "text": text_content,
            "source": source
            
        }
        entities.append(entity)

    # Check if the collection already contains data
    if collection.num_entities == 0:
        collection.insert(entities)
        print(f"Inserted {len(entities)} entities into the collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already contains data. Skipping insertion.")
    return collection

# Function to format documents with their sources and extract associated images
def format_docs(docs):
    formatted_content = ""
    sources = set() 
    
    # Loop through each document to retrieve text and source
    for doc in docs:
        content = getattr(doc, "text", "")
        source = doc.metadata.get("source", "Unknown source")

        formatted_content += f"{content}\n\n"
        sources.add(source) 

    # Combine sources into a formatted string
    formatted_sources = "\n".join(sources)

    return formatted_content, formatted_sources

# Function to invoke the language model for generating a response
def invoke_llm_for_response(query: str):
    api_key = get_api_key() 
    if not isinstance(query, str):
        raise ValueError("The input query must be a string.")
    
    if len(query.split()) < 2:  
        return "Please ask a more specific question.", [], []  # Ensure this return has three items
    
    # Initialize the language model
    llm = ChatMistralAI(model='open-mistral-7b', api_key=api_key)

    # Define the prompt template
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provide answers to questions by using fact-based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Assistant:"""

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # Ensure `texts` are strings
    texts = get_texts_data(corpus_source)
    if not texts:
        return "No content found in the specified URLs. Please check your data source.", [], []
    texts = [text['page_content'] if isinstance(text, dict) and 'page_content' in text else text for text in texts if isinstance(text, str) or isinstance(text, dict)]

    # Define the fields and search parameters for the Milvus retriever
    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}
    collection = initialize_milvus()

    # Initialize the Milvus retriever
    retreiver = MilvusCollectionHybridSearchRetriever(
        collection=collection,
        rerank=WeightedRanker(0.7, 0.3),
        anns_fields=[dense_field, sparse_field],
        field_embeddings=[dense_embedding_func, sparse_embedding_func],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=5,
        text_field=text_field,
    )

    hybrid_results = retreiver.invoke(query)
    # Have to implement re-ranking function for the hybrid retriever for exact query matching
    formatted_content, formatted_sources = format_docs(hybrid_results)

    context_callable = lambda x: formatted_content

    # Define the RAG chain manually with the specified format
    rag_chain = (
        {"context": context_callable, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the RAG chain with the specific question
    response = rag_chain.invoke({"input": query})

    final_response = f"{response}\n\nSources:\n{formatted_sources}"

    return final_response,formatted_sources

