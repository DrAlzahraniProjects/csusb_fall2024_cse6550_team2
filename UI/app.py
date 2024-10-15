import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import pipeline
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App Configuration
st.set_page_config(page_title="Academic Chatbot - Team2", layout="wide")

# CSS Styling
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



# Initialize Session State for Conversation History
if 'conversation' not in st.session_state:  
    st.session_state['conversation'] = []
if 'input_given' not in st.session_state:
    st.session_state['input_given'] = False
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Display an Animated Typing Title
def typing_title_animation(title, delay=0.3):
    placeholder = st.empty()
    words = title.split()
    full_text = ""
    for word in words:
        full_text += word + " "
        placeholder.markdown(f"<h1 style='text-align: center;'>{full_text.strip()}</h1>", unsafe_allow_html=True)
        time.sleep(delay)
    return placeholder

# Function to process user input and generate bot response using Milvus
def process_input():
    user_input = st.session_state['user_input']

    # Append user input to conversation
    st.session_state['conversation'].append({"role": "user", "content": user_input})

    # Process the input using Milvus and Hugging Face QA pipeline
    with st.spinner('Processing your question...'):
        try:
            # Encode the question
            query_embedding = embedding_model.encode([user_input]).tolist()

            # Define search parameters for Milvus
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # Search in the Milvus collection
            results = collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=5,
                expr=None
            )

            # Check if any results found
            if not results or len(results[0]) == 0:
                bot_reply = "Sorry, I cannot answer that now."
            else:
                # Aggregate retrieved texts as context for the QA model
                context = " ".join([hit.entity.get("text") for hit in results[0]])

                # Use the QA pipeline to generate an answer based on context
                try:
                    bot_reply = qa_pipeline_model(question=user_input, context=context)['answer']
                except Exception as e:
                    bot_reply = "Sorry, I encountered an error while processing your request."

        except Exception as e:
            bot_reply = "Sorry, I cannot answer that now."

    # Append the bot reply to the conversation history
    st.session_state['conversation'].append({"role": "bot", "content": bot_reply})

    # Clear the input field after processing
    st.session_state['user_input'] = ''  # This will reset the text input field

# Placeholder for the Animated Title
if 'title_placeholder' not in st.session_state:
    st.session_state['title_placeholder'] = st.empty()

# Function to Clear Animated Title After Input
def clear_title():
    st.session_state['title_placeholder'].empty()

# Display Animated Title if No Input Yet
if not st.session_state.get('input_given', False):
    st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
else:
    clear_title()
    # Display the Fixed Title at the Top Left with a Logo
    st.markdown("""
        <div style="position: fixed; top: 10px; left: 10px; font-size: 24px; font-weight: bold;">
            Academic Chatbot
        </div>
    """, unsafe_allow_html=True)

# Display Conversation History
for i, message in enumerate(st.session_state['conversation']):
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)
        # Add Like/Dislike buttons under the bot's message
        col1, col2 = st.columns(2)
        with col1: #WARNING ethe i for the button will always be an odd number
            if st.button(f'👍', key=f'like_{i}'):
                st.write('Liked!')
        with col2:
            if st.button(f'👎', key=f'dislike_{i}'):
                st.write('Disliked.')

# Text Input
st.text_input("You:", key="user_input", placeholder="Ask me anything academic...", label_visibility='collapsed', on_change=process_input)



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

# Get Environment Variables
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
milvus_host = os.getenv('MILVUS_HOST', 'standalone')  # Use environment variable or default to 'standalone'
milvus_port = int(os.getenv('MILVUS_PORT', 19530))    # Use environment variable or default to 19530

# Data
web_file= ["https://catalog.csusb.edu/graduate-degree-programs/"]
def split_pages_by_documents():
    loader = WebBaseLoader(web_paths=web_file)
    documents = loader.load()
    # Create a text splitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split the text into chunks of 1000 characters
        chunk_overlap=300,  # Overlap the chunks by 300 characters
        is_separator_regex=False,  # Don't split on regex
    )
    # Split the documents into chunks
    docs = text_splitter.split_documents(documents)
    return docs
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def insert_vectors():
    docs = split_pages_by_documents()
    texts = [doc.page_content for doc in docs]
    vectors = model.encode(texts)
    # Prepare data for insertion
    data = [
        vectors.tolist(),  # embedding field
        texts  # text field
    ]
    print(f"Inserted {len(vectors)} vectors into the collection.")
    print(data)
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
        st.stop()  # Stop the execution if Milvus connection fails
    
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

# Check if the collection is initialized before proceeding
collection = initialize_milvus()
if collection is None:
    st.error("Unable to initialize Milvus. The chatbot cannot proceed without the database connection.")

# Initialize Models
@st.cache_resource
def initialize_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline_model = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    return embedding_model, qa_pipeline_model

embedding_model, qa_pipeline_model = initialize_models()

# Function to insert data into the Milvus collection
def insert_data(collection, model):
    docs = split_pages_by_documents()
    texts = [doc.page_content for doc in docs]

    # Prepare lists to hold truncated texts and embeddings
    all_truncated_texts = []
    all_embeddings = []

    # Process each text
    for text in texts:
        # Truncate the text to the max length of 500 characters
        truncated_text = text[:500]  # Truncating to 500 characters
        all_truncated_texts.append(truncated_text)

    # Encode all truncated texts to get embeddings
    embeddings = model.encode(all_truncated_texts).tolist()

    # Prepare data for insertion
    data = [
        all_truncated_texts,
        embeddings
    ]
    print(f"Inserted {len(all_truncated_texts)} vectors into the collection.")
    
    # Insert data into Milvus
    collection.insert(data)
    st.info(f"Inserted {len(all_truncated_texts)} records into '{collection.name}'.")