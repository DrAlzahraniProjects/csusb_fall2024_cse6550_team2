import streamlit as st
import time
import os
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Web Scraping Function for CSUSB CSE webpage
def scrape_csusb_cse_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join([p.text for p in soup.find_all("p")])
        return text
    else:
        return "Error: Unable to retrieve data."

# Scrape the CSUSB CSE page and create LangChain documents
cse_page_url = "https://www.csusb.edu/cse"
webpage_content = scrape_csusb_cse_website(cse_page_url)

# Convert the webpage content into LangChain Document
doc = Document(page_content=webpage_content)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents([doc])

# Initialize Milvus and OpenAI Embeddings
embeddings = OpenAIEmbeddings()
milvus = Milvus(embedding_function=embeddings, collection_name="csusb_cse_collection")
milvus.add_texts(texts)

# Create the RAG Chain for retrieval
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=milvus.as_retriever(),
    return_source_documents=True
)

# Streamlit App Configuration
st.set_page_config(page_title="Academic Chatbot - Team2")

# CSS Styling
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for animated typing title
def typing_title_animation(title, delay=0.3):
    placeholder = st.empty()
    words = title.split()
    full_text = ""
    for word in words:
        full_text += word + " "
        placeholder.markdown(f"<h1 style='text-align: center;'>{full_text.strip()}</h1>", unsafe_allow_html=True)
        time.sleep(delay)
    return placeholder

# Function to display rating buttons
def display_rating_buttons(index):
    st.markdown(f"""
        <div class="rating-buttons">
            <span class="rating-icon" title="Like">👍</span>
            <span class="rating-icon" title="Dislike">👎</span>
        </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'input_given' not in st.session_state:
    st.session_state['input_given'] = False

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Function to handle RAG-based chatbot responses
def chatbot_response_rag(user_input):
    query = user_input.lower()
    result = rag_chain(query)
    return result['result'], result['source_documents']

# Function to process user input and get bot response
def process_input():
    user_input = st.session_state['user_input']
    st.session_state['conversation'].append({"role": "user", "content": user_input})

    # Get response from the RAG-based chatbot
    bot_reply, source_documents = chatbot_response_rag(user_input)
    
    # Add the bot's response and source to the conversation
    st.session_state['conversation'].append({"role": "bot", "content": bot_reply})
    
    # Optionally, you could display the sources of the response:
    for doc in source_documents:
        st.markdown(f"Source: {doc.metadata['source']}")

    st.session_state['user_input'] = ''
    st.session_state['input_given'] = True

# Sidebar for chat statistics
st.sidebar.title("Metric Summary")
# Add various expander sections as in your original code
with st.sidebar.expander("Number of questions"):
    st.write("Details go here...")

# Placeholder for the animated title
if 'title_placeholder' not in st.session_state:
    st.session_state['title_placeholder'] = st.empty()

# Show animated title if no input is given
if not st.session_state['input_given']:
    st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
else:
    st.session_state['title_placeholder'].empty()
    st.markdown(f"""
        <div class="fixed-logo-text">Academic Chatbot</div>
    """, unsafe_allow_html=True)

# Display conversation history
for index, message in enumerate(st.session_state['conversation']):
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)
        display_rating_buttons(index)

# Input box
st.text_input(
    "You: ",
    key="user_input",
    placeholder="Ask me anything academic...",
    on_change=process_input,
    label_visibility="collapsed",
)
