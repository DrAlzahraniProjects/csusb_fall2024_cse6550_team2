import os
import streamlit as st
# from pymilvus import MilvusClient, model, connections, db
import requests
import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from mistralai import Mistral
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Milvus
# from langchain.embeddings import MistralEmbeddings
# from langchain.llms import Mistral
from langchain.chains import RetrievalQA
from pymilvus import connections, Collection
import numpy as np
# from langchain_community.llms import Mistral
import streamlit as st
import torch
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mistral_llm import generate_response

# Connect to Milvus server
def create_connection():
    connections.connect(alias="default", host='localhost', port='19530')  # Adjust host and port to your setup
    print("Connected to Milvus")

create_connection()
# Define the collection schema
def create_collection():
    collection_name = "aca_database"

    if utility.has_collection(collection_name):
        # If collection exists, load it
        collection = Collection(collection_name)
        #print(f"Collection '{collection_name}' loaded.")
    else:
        # Define fields for a new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust dimension based on Mistral
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000)  # Storing webpage content
        ]

        schema = CollectionSchema(fields, description="Collection for webpage embeddings")

        # Create a new collection
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection created: {collection.name}")

    return collection  # Return the collection object



collection = create_collection()

# Actual app.py

# Changes tab title (Warning: Leave at top)
st.set_page_config(page_title = "Academic Chatbot - Team2")

# CSS styling
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for chatbot responses
def chatbot_response(user_input):
    return "I'm sorry, I don't have an answer for that right now."

# Function to process user input and generate bot response
def process_input():
    user_input = st.session_state['user_input']
    st.session_state['conversation'].append({"role": "user", "content": user_input})
    bot_reply = chatbot_response(user_input)
    st.session_state['conversation'].append({"role": "bot", "content": bot_reply})
    st.session_state['user_input'] = ''
    st.session_state['input_given'] = True  # Mark that input has been given

# Function for animated typing title
def typing_title_animation(title, delay=0.3):
    placeholder = st.empty()  # Create a placeholder to update dynamically
    words = title.split()
    full_text = ""
    for word in words:
        full_text += word + " "
        placeholder.markdown(f"<h1 style='text-align: center;'>{full_text.strip()}</h1>", unsafe_allow_html=True)
        time.sleep(delay)
    return placeholder

# Function to display rating buttons for each bot response
def display_rating_buttons(index):
    st.markdown(f"""
        <div class="rating-buttons">
            <span class="rating-icon" title="Like">👍</span>
            <span class="rating-icon" title="Dislike">👎</span>
        </div>
    """, unsafe_allow_html=True)

# Apply the external CSS file
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for input tracking
if 'input_given' not in st.session_state:
    st.session_state['input_given'] = False  # Track if input has been given

# Sidebar for chat history and statistics
st.sidebar.title("Metric Summary")

# Number of questions
with st.sidebar.expander("Number of questions"):
    st.write("Details go here...")

# Number of correct answers
with st.sidebar.expander("Number of correct answers"):
    st.write("Details go here...")

# Number of incorrect answers
with st.sidebar.expander("Number of incorrect answers"):
    st.write("Details go here...")

# User engagement metrics
with st.sidebar.expander("User engagement metrics"):
    st.write("Details go here...")

# Response time analysis
with st.sidebar.expander("Response time analysis"):
    st.write("Details go here...")

# Accuracy rate
with st.sidebar.expander("Accuracy rate"):
    st.write("Details go here...")

# Common topics or keywords
with st.sidebar.expander("Common topics or keywords"):
    st.write("Details go here...")

# User satisfaction ratings
with st.sidebar.expander("User satisfaction ratings"):
    st.write("Details go here...")

# Improvement over time
with st.sidebar.expander("Improvement over time"):
    st.write("Details go here...")

# Statistics per day and overall
with st.sidebar.expander("Statistics per day and overall"):
    st.write("Details go here...")

# Feedback summary
with st.sidebar.expander("Feedback summary"):
    st.write("Details go here...")

# Placeholder for the animated title
if 'title_placeholder' not in st.session_state:
    st.session_state['title_placeholder'] = st.empty()

# Animate the title if no input has been given
if not st.session_state['input_given']:
    st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
else:
    # Clear the animated title once input is given
    st.session_state['title_placeholder'].empty()

    # Display the fixed title at the top left with a logo
    st.markdown(f"""
        <div class="fixed-logo-text">Academic Chatbot</div>
    """, unsafe_allow_html=True)

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
    with st.spinner("Initializing, Please Wait..."):
        vector_store = connect_milvus()

# Display conversation history
for index, message in enumerate(st.session_state['conversation']):
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
        st.caption(f":blue[{message['source']}]")
        # Display the rating buttons below each bot response
        display_rating_buttons(index)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)

# Input box
# st.text_input(
#     "You: ",
#     key="user_input",
#     placeholder="Ask me anything academic...",
#     on_change=process_input,
#     label_visibility="collapsed",
# )

 # Handle user input
    if prompt := st.chat_input("Message Team2 academic chatbot"):      
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

        response_placeholder = st.empty()

        with response_placeholder.container():
            with st.spinner('Generating Response'):
                # generate response from LLM 
                answer, source = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
            response_placeholder.markdown(f"""
                <div class='assistant-message'>
                    {answer}
                </div>
            """, unsafe_allow_html=True)
        st.caption(f":blue[{source}]")

        # Add like and dislike buttons for the newly generated assistant message
        st.markdown("""
            <div class='feedback-buttons'>
                <button aria-label="👍 Like" onclick="window.location.reload()">👍</button>
                <button aria-label="👎 Dislike" onclick="window.location.reload()">👎</button>
            </div>
        """, unsafe_allow_html=True)