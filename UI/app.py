from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility
from transformers import pipeline
import torch
from dotenv import load_dotenv
import sys
import os

# Load environment variables from the .env file (optional)
load_dotenv()
print("Hello")
# Get the token from the environment variable
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

# Initialize Milvus connection
milvus_host = os.getenv('MILVUS_HOST', 'standalone')  # Use environment variable or default to 'standalone'
milvus_port = int(os.getenv('MILVUS_PORT', 19530))    # Use environment variable or default to 19530
milvus_host_local = 'localhost'


def setup_milvus():
    # Connect to Milvus server
    connections.connect(host=milvus_host, port=milvus_port)

    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    # Create schema
    schema = CollectionSchema(fields, description="Chatbot data collection")

    # Create collection if not exists
    collection_name = "chatbot_collection"
    if collection_name not in connections.list_collections():
        collection = Collection(name=collection_name, schema=schema)
        # Create an index for the embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 100}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Collection '{collection_name}' created and indexed.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    # Load the collection to memory
    collection.load()
    return collection

def insert_data(collection, texts, model):
    # Encode texts to get embeddings
    embeddings = model.encode(texts).tolist()

    # Prepare data for insertion
    data = [
        texts,
        embeddings
    ]

    # Insert data into Milvus
    collection.insert(data)
    print(f"Inserted {len(texts)} records into '{collection.name}'.")

def chatbot():
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline_model = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad"
    )

    print("Welcome to the Milvus-Transformers Chatbot!")
    print("Type 'exit' to end the conversation.\n")

    while True:
        # Get user input
        question = input("You: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Encode the question
        query_embedding = embedding_model.encode([question]).tolist()

        # Define search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # Perform similarity search
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=5,
            expr=None
        )

        # Check if any results found
        if not results or len(results[0]) == 0:
            print("Chatbot: I'm sorry, I don't have an answer for that.\n")
            continue

        # Aggregate retrieved texts as context
        context = " ".join([hit.entity.get("text") for hit in results[0]])

        # Use the QA pipeline to generate an answer
        try:
            answer = qa_pipeline_model(question=question, context=context)['answer']
            print(f"Chatbot: {answer}\n")
        except Exception as e:
            print(f"Chatbot: Sorry, I encountered an error: {e}\n")

if __name__ == "__main__":
    # Setup Milvus and get the collection
    collection = setup_milvus()

    # Example data insertion (Run only once or as needed)
    # Uncomment the following lines to insert data
    """
    example_texts = [
        "Transformers are a type of neural network architecture that has revolutionized NLP.",
        "Milvus is a high-performance vector database for similarity search.",
        "Chatbots can leverage machine learning models to generate human-like responses.",
        "The Transformers library by Hugging Face provides pre-trained models for various NLP tasks.",
        "Sentence transformers are used to generate meaningful sentence embeddings."
    ]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    insert_data(collection, example_texts, model)
    """

    # Start the chatbot
    try:
        chatbot()
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye!")
        sys.exit(0)
