import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility
from transformers import pipeline
import torch
from dotenv import load_dotenv

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
connections.connect(host=milvus_host, port=milvus_port)

# Load Mistral embeddings using SentenceTransformer
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load Mistral LLM from Hugging Face
llm_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad"
    )

# Initialize Milvus client
milvus_client = connections.connect(host=milvus_host, port=milvus_port)

# Sidebar with user input
user_question = st.text_input('Ask your question here:')

if user_question:
    with st.spinner('Processing your question...'):
        # Generate embeddings for the question using the embedding model
        question_embedding = embedding_model.encode([user_question])

        # Search in Milvus for relevant documents
        search_params = {
            'collection_name': 'academic_chatbot',  # Replace with your Milvus collection name
            'query_emb': question_embedding[0],
            'top_k': 5  # Number of documents to retrieve
        }
        search_results = utility.search_vectors(**search_params)

        # Display search results
        st.subheader('Search Results:')
        for result in search_results:
            st.write(f"- {result['document']}")

        # Use the Mistral LLM model for chatbot response generation
        chatbot_response = llm_pipeline(user_question, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.subheader('Chatbot Response:')
        st.write(chatbot_response)

        # Placeholder for like and unlike buttons
        st.subheader('Feedback:')
        if st.button('Like'):
            st.write('Liked!')
        if st.button('Unlike'):
            st.write('Disliked.')

# Disconnect from Milvus
connections.disconnect()

# Changes tab title (Warning: Leave at top)
st.set_page_config(page_title="Academic Chatbot - Team2-Updated")

# CSS styling
with open("./assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for chatbot responses
def chatbot_response(user_input):
    responses = {
        'hi': 'Hello! How can I support you with your academic goals today?',
        'hello': 'Hi there! What academic assistance do you need right now?',
        'bye': 'Goodbye! Don’t hesitate to return if you have more questions.',
        'what can you do': 'I can assist you with academic advising, research topics, and provide study tips. How can I help you?',
        'help': 'Absolutely! What specific academic challenges are you facing?'
    }

    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return responses[key]
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
with open("./assets/style.css") as f:
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

# Display conversation history
for index, message in enumerate(st.session_state['conversation']):
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)

        # Display the rating buttons below each bot response
        display_rating_buttons(index)

# Input box
st.text_input(
    "You: ",
    key="user_input",
    placeholder="Ask me anything academic...",
    on_change=process_input,
    label_visibility="collapsed",
)
