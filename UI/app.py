import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import pipeline
import os
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from the .env file
load_dotenv()

# Streamlit App Configuration
st.set_page_config(page_title="Academic Chatbot - Team2", layout="wide")

# Initialize Session State for Conversation History
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# CSS Styling
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar: Stats Section
st.sidebar.header("Statistics")
st.sidebar.markdown("Number of questions: N")
st.sidebar.markdown("Number of correct answers: N")
st.sidebar.markdown("Number of incorrect answers: N")
st.sidebar.markdown("User engagement metrics: ...")
st.sidebar.markdown("Response time analysis: ...")
st.sidebar.markdown("Accuracy rate: ...")
st.sidebar.markdown("Common topics or keywords: ...")
st.sidebar.markdown("User satisfaction ratings: ...")
st.sidebar.markdown("Improvement over time: ...")
st.sidebar.markdown("Feedback summary: ...")
st.sidebar.markdown("Statistics per day and overall: ...")

# Centered Chat Bar for User Input
st.markdown("<h2 style='text-align: center;'>Ask Your Question</h2>", unsafe_allow_html=True)
user_question = st.text_input("You:", key="user_input", placeholder="Ask me anything academic...", label_visibility='collapsed')

# Get Environment Variables
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
milvus_host = os.getenv('MILVUS_HOST', 'standalone')  # Use environment variable or default to 'standalone'
milvus_port = int(os.getenv('MILVUS_PORT', 19530))    # Use environment variable or default to 19530

# Data
web_file = ["https://catalog.csusb.edu/graduate-degree-programs/"]
def split_pages_by_documents():
    loader = WebBaseLoader(web_paths=web_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300, 
        is_separator_regex=False
    )
    docs = text_splitter.split_documents(documents)
    return docs

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def insert_vectors():
    docs = split_pages_by_documents()
    texts = [doc.page_content for doc in docs]
    vectors = model.encode(texts)
    data = [vectors.tolist(), texts]
    return data

# Initialize Milvus Connection and Collection
@st.cache_resource
def initialize_milvus():
    try:
        connections.connect(host=milvus_host, port=milvus_port)
        st.success(f"Connected to Milvus at {milvus_host}:{milvus_port}")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        st.error("Error details:")
        st.text(traceback.format_exc())  # Show full traceback
        st.stop()
    
    collection_name = "academic_chatbot"
    if collection_name not in utility.list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, description="Chatbot data collection")
        collection = Collection(name=collection_name, schema=schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 100}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        st.info(f"Collection '{collection_name}' created and indexed.")
    else:
        collection = Collection(name=collection_name)
        st.info(f"Collection '{collection_name}' already exists.")
    
    collection.load()
    return collection

collection = initialize_milvus()

# Initialize Models
@st.cache_resource
def initialize_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline_model = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad",
        use_auth_token=huggingface_token
    )
    return embedding_model, qa_pipeline_model

embedding_model, qa_pipeline_model = initialize_models()

# Insert Data into Milvus Collection
def insert_data(collection, model):
    docs = split_pages_by_documents()
    texts = [doc.page_content for doc in docs]

    all_truncated_texts = []
    all_embeddings = []

    for text in texts:
        truncated_text = text[:500]
        all_truncated_texts.append(truncated_text)

    embeddings = model.encode(all_truncated_texts).tolist()

    data = [all_truncated_texts, embeddings]
    collection.insert(data)
    st.info(f"Inserted {len(all_truncated_texts)} records into '{collection.name}'.")

# Handle User Question and Generate Response
if user_question:
    st.session_state['conversation'].append({"role": "user", "content": user_question})
    with st.spinner('Processing your question...'):
        query_embedding = embedding_model.encode([user_question]).tolist()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=5,
            expr=None
        )

        if not results or len(results[0]) == 0:
            answer = "I'm sorry, I don't have an answer for that."
        else:
            context = " ".join([hit.entity.get("text") for hit in results[0]])
            try:
                answer = qa_pipeline_model(question=user_question, context=context)['answer']
            except Exception as e:
                answer = f"Sorry, I encountered an error: {e}"

        st.session_state['conversation'].append({"role": "bot", "content": answer})

# Display Conversation History
st.subheader("Conversation")
for message in st.session_state['conversation']:
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)

# Feedback Buttons
if user_question:
    st.subheader('Feedback:')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Like'):
            st.write('Liked!')
    with col2:
        if st.button('Unlike'):
            st.write('Disliked.')
