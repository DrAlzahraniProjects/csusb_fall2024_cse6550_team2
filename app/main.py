# Imports
from web_crawler import initialize_and_scrape
from utils import initialize_metrics_sidebar, initialize_session_state, update_metrics, reset_metrics, typing_title_animation, update_likes, update_dislikes, handle_feedback
import streamlit as st
import os
import time
from pymilvus import MilvusException
from backend import *


# Page configuration
st.set_page_config(page_title="Academic Chatbot - Team2")
# Prompt for API key if not already provided
if "api_key" not in st.session_state:
    api_key = os.environ.get("API_KEY")
    if api_key:
        st.session_state["api_key"] = api_key
        os.environ["API_KEY"] = api_key
else:
    api_key = st.session_state["api_key"]

# Initialize session state variables
initialize_session_state()

# Initialize metrics and placeholders in sidebar
initialize_metrics_sidebar()

# Proceed only if API key is set
if "API_KEY" in os.environ:

   # Ensure `initialize_and_scrape()` runs only once
    if 'milvus_initialized' not in st.session_state:
        with st.spinner("Initializing, Please Wait..."):
            initialize_and_scrape()
        st.session_state['milvus_initialized'] = True  # Mark as initialized

    # Typing animation for title
    if not st.session_state['input_given'] and not st.session_state['title_animated']:
        st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.1)
        st.session_state['title_animated'] = True
    else:
        st.markdown(f"""<div class="chat-title">Academic Advisor Chatbot</div>""", unsafe_allow_html=True)

    # Load CSS styling
    with open("./style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Function to process user input and generate bot response
    def process_input(prompt):
        st.session_state['num_questions'] += 1
        st.session_state['messages'].append({"role": "user", "content": prompt})

        start_time = time.time()
        with st.spinner('Generating Response, Please Wait...'):
            try:
                response = invoke_llm_for_response(prompt)
            except MilvusException as e:
                response = "Error: Query format issue. Try a more detailed question." if "vector type must be the same" in str(e) else f"Error: {e}"
            
            # Append assistant's response to messages
            st.session_state['messages'].append({"role": "assistant", "content": {"response": response}})
            st.session_state['total_response_time'] += time.time() - start_time
            st.session_state['num_responses'] += 1
            update_metrics()  # Update metrics after generating a response

    # Handle user input in chat
    if prompt := st.chat_input("Message Team2 academic chatbot"):
        process_input(prompt)

# Display chat messages and feedback
    for index, message in enumerate(st.session_state.get('messages', [])):
        if message['role'] == 'user':
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message['content'].get("response", ""),unsafe_allow_html=True)

            # Use st.feedback with "thumbs" option for thumbs-up and thumbs-down feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{index}",
                on_change=handle_feedback,
                args=(index,)
            )




    # Sidebar Reset Button
    if st.sidebar.button("Reset Metrics"):
        reset_metrics()  # Reset all metrics and refresh the sidebar with zeroed values

else:
    st.warning("API key is required to proceed.")
