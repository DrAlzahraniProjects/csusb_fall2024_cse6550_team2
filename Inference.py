from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from langchain_community.embeddings import HuggingFaceEmbeddings  # Adjust this import if necessary
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
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
    utility
)
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.utils.sparse import BM25SparseEmbedding
import nltk
import os
from langchain.text_splitter import CharacterTextSplitter
from urllib.parse import urljoin

nltk.download('punkt')
MILVUS_URI = "./milvus_lite/milvus_vector.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
collection_name = "Academic_Webpages"

# Create a folder for saving images if it doesn't exist
os.makedirs("downloaded_images", exist_ok=True)

web_pages = [
    "https://www.csusb.edu/cse",
    "https://catalog.csusb.edu/"
]

# Data Scraping
def load_webpages(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')

        content_list = []
        image_paths = []
        for p in paragraphs:
            paragraph_text = p.get_text()
            # Extract all links within the paragraph
            links = [a['href'] for a in p.find_all('a', href=True)]
            # Combine the paragraph text with its links
            if links:
                combined_text = f"{paragraph_text} (Links: {', '.join(links)})"
            else:
                combined_text = paragraph_text
            content_list.append(combined_text)

        # Download all images present in the page
        images = soup.find_all('img')
        for img in images:
            img_src = img.get('src')
            if img_src:
                img_url = urljoin(url, img_src)
                img_path = save_image(img_url)
                if img_path:
                    image_paths.append(img_path)
                save_image(img_url)
     # Combine content and associate it with image paths
        content = " ".join(content_list)
        return {"content": content, "source": url, "images": image_paths}           

        # return " ".join(content_list)
    else:
        print(f"Failed to retrieve {url}")
        # return ""
        return {"content": "", "source": url, "images": []}

# Function to download and save images locally
def save_image(img_url):
    try:
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            # Create a filename from the image URL
            img_name = img_url.split("/")[-1]
            img_path = os.path.join("downloaded_images", img_name)
            with open(img_path, "wb") as img_file:
                img_file.write(img_response.content)
            print(f"Saved image: {img_path}")
            return img_path  # Return the path of the saved image
        else:
            print(f"Failed to download image from {img_url}")
    except Exception as e:
        print(f"Error saving image from {img_url}: {e}")
    return None



# Function to download and save images locally and return the path
def save_image(img_url):
    try:
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            # Create a filename from the image URL
            img_name = img_url.split("/")[-1]
            img_path = os.path.join("downloaded_images", img_name)
            with open(img_path, "wb") as img_file:
                img_file.write(img_response.content)
            print(f"Saved image: {img_path}")
            return img_path  # Return the path of the saved image
        else:
            print(f"Failed to download image from {img_url}")
    except Exception as e:
        print(f"Error saving image from {img_url}: {e}")
    return None

# Data Scraping with image links
def load_webpages(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')

        content_list = []
        image_paths = []

        for p in paragraphs:
            paragraph_text = p.get_text()
            links = [a['href'] for a in p.find_all('a', href=True)]
            if links:
                combined_text = f"{paragraph_text} (Links: {', '.join(links)})"
            else:
                combined_text = paragraph_text
            content_list.append(combined_text)

        # Download all images present on the page and store their paths
        images = soup.find_all('img')
        for img in images:
            img_src = img.get('src')
            if img_src:
                img_url = urljoin(url, img_src)
                img_path = save_image(img_url)
                if img_path:
                    image_paths.append(img_path)

        # Combine content and associate it with image paths
        content = " ".join(content_list)
        return {"content": content, "source": url, "images": image_paths}
    else:
        print(f"Failed to retrieve {url}")
        return {"content": "", "source": url, "images": []}

# Split text into smaller chunks
def split_text(text, chunk_size=1500):
    text_splitter = CharacterTextSplitter(separator=",", chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)

# Get the texts data from the web pages with cleaning
def get_texts_data():
    texts = []

    for url in web_pages:
        page_data = load_webpages(url)
        content = page_data["content"]
        images = page_data["images"]
        source = page_data["source"]

        if content:
            # Clean up newline characters in the content
            cleaned_content = content.replace("\n", " ")
            # Split the cleaned content into smaller chunks
            split_contents = split_text(cleaned_content)

            # Create a document entry with each chunk, the source, and associated images
            for split_content in split_contents:
                texts.append({
                    "page_content": split_content,
                    "source": source,
                    "images": images
                })

    print(texts)  # For debugging, this will show the cleaned texts with sources and images
    return texts

# Call the function to get cleaned and split texts

# Get the texts data (returns a list of dictionaries with content, source, and images)
texts = get_texts_data()
def extract_text_content(texts):
    return [text["page_content"] for text in texts if "page_content" in text]

# Extract the cleaned text content from the dictionaries for embedding
text_contents = extract_text_content(texts)

# Initialize the dense and sparse embeddings
dense_embedding_func = HuggingFaceEmbeddings(model_name=MODEL_NAME)
dense_dim = len(dense_embedding_func.embed_query(text_contents[1]))
print(dense_dim)
sparse_embedding_func = BM25SparseEmbedding(corpus=text_contents)
sparse_embedding_func.embed_query(text_contents[1])
# print(sparse_embedding_func)

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
    for text in text_contents:
        entity = {
            "dense_vector": dense_embedding_func.embed_documents([text])[0],
            "sparse_vector": sparse_embedding_func.embed_documents([text])[0],
            "text": text,
        }
        entities.append(entity)

    # Check if the collection already contains data
    if collection.num_entities == 0:
        collection.insert(entities)
        print(f"Inserted {len(entities)} entities into the collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already contains data. Skipping insertion.")
    return collection

def retreiver():
  # Ensure `texts` are strings
    texts = text_contents
    texts = [text['page_content'] if isinstance(text, dict) and 'page_content' in text else text for text in texts if isinstance(text, str) or isinstance(text, dict)]

    dense_embedding_func = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    sparse_embedding_func = BM25SparseEmbedding(corpus=texts)

    collection = initialize_milvus()

    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}

    Hybridretriever = MilvusCollectionHybridSearchRetriever(
        collection=collection,
        rerank=WeightedRanker(0.5, 0.5),
        anns_fields=[dense_field, sparse_field],
        field_embeddings=[dense_embedding_func, sparse_embedding_func],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=3,
        text_field=text_field,
    )
    return Hybridretriever

# Function to format documents with their sources and extract associated images
def format_docs(docs):
    formatted_content = ""
    sources = set()  # Use a set to avoid duplicate sources
    images = []

    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown source')
        image_paths = doc.metadata.get('images', [])

        formatted_content += f"{content}\n\n"
        sources.add(source)  # Collect unique sources
        images.extend(image_paths)

    # Combine sources into a formatted string
    formatted_sources = "\n".join(sources)

    return formatted_content, formatted_sources, images

def invoke_llm_for_response(query: str):
    os.environ["MISTRAL_API_KEY"] = "IetRnH5Lb578MdB5Ml0HNTdMBzeHUe7q"
    if not isinstance(query, str):
        raise ValueError("The input query must be a string.")
    
    # Initialize the language model
    llm = ChatMistralAI(model='open-mistral-7b', ap_key=os.environ["MISTRAL_API_KEY"])

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
    texts = get_texts_data()
    texts = [text['page_content'] if isinstance(text, dict) and 'page_content' in text else text for text in texts if isinstance(text, str) or isinstance(text, dict)]

    dense_embedding_func = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    sparse_embedding_func = BM25SparseEmbedding(corpus=texts)

    collection = initialize_milvus()

    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "IP", "params": {}}

    retreiver = MilvusCollectionHybridSearchRetriever(
        collection=collection,
        rerank=WeightedRanker(0.5, 0.5),
        anns_fields=[dense_field, sparse_field],
        field_embeddings=[dense_embedding_func, sparse_embedding_func],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=3,
        text_field=text_field,
    )

    hybrid_results = retreiver.invoke(query)
    formatted_docs, sources, images = format_docs(hybrid_results)
    print(formatted_docs, sources, images,"Formatted Docs")

    context_callable = lambda x: formatted_docs

    # Define the RAG chain manually with the specified format
    rag_chain = (
        {"context": context_callable, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the RAG chain with the specific question
    response = rag_chain.invoke({"input": query})
    print(response, "Response Generated")
    return response, sources, images