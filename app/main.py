# Imports
from WebCrawler import initialize_and_scrape
from utils import typing_title_animation,reset_metrics
import streamlit as st
import os
import time
from pymilvus import MilvusException
from backend import *
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Academic Chatbot - Team2")

# Prompt for API key if not already provided
if "api_key" not in st.session_state:
    api_key = st.text_input("Please enter your Mistral API key:", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
        os.environ["API_KEY"] = api_key
else:
    api_key = st.session_state["api_key"]

# Proceed only if API key is set
if "API_KEY" in os.environ:
    if 'conversation' not in st.session_state:
        with st.spinner("Initializing, Please Wait..."):
            initialize_and_scrape()
    # Initialize session state variables
    for key, default_value in [
        ('input_given', False), ('title_animated', False), ('title_placeholder', st.empty()),
        ('num_questions', 0), ('num_correct_answers', 0), ('num_incorrect_answers', 0),
        ('num_responses', 0), ('user_engagement', {'likes': 0, 'dislikes': 0}),
        ('rated_responses', {}), ('y_true', []), ('y_pred', []), ('conversation', []),
        ('messages', []), ('total_response_time', 0)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Typing animation for title
    if not st.session_state['input_given'] and not st.session_state['title_animated']:
        st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
        st.session_state['title_animated'] = True
    else:
        st.markdown(f"""<div class="chat-title">Academic Advisor Chatbot</div>""", unsafe_allow_html=True)

    # Load CSS styling
    with open("./style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Functions for updating metrics based on user feedback
    def update_likes(index):
        previous_rating = st.session_state['rated_responses'].get(index)
        if previous_rating != 'liked':
            if previous_rating == 'disliked':
                st.session_state['num_incorrect_answers'] -= 1
                st.session_state['user_engagement']['dislikes'] -= 1
            st.session_state['num_correct_answers'] += 1
            st.session_state['user_engagement']['likes'] += 1
            st.session_state['rated_responses'][index] = 'liked'

    def update_dislikes(index):
        previous_rating = st.session_state['rated_responses'].get(index)
        if previous_rating != 'disliked':
            if previous_rating == 'liked':
                st.session_state['num_correct_answers'] -= 1
                st.session_state['user_engagement']['likes'] -= 1
            st.session_state['num_incorrect_answers'] += 1
            st.session_state['user_engagement']['dislikes'] += 1
            st.session_state['rated_responses'][index] = 'disliked'

    def display_rating_buttons(index):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("👍 Like", key=f"like_button_{index}", on_click=update_likes, args=(index,))
        with col2:
            st.button("👎 Dislike", key=f"dislike_button_{index}", on_click=update_dislikes, args=(index,))

    # Function to process user input and generate bot response
    def process_input(prompt):
        st.session_state['num_questions'] += 1
        st.session_state['messages'].append({"role": "user", "content": prompt})

        start_time = time.time()
        with st.spinner('Generating Response...'):
            try:
                response = invoke_llm_for_response(prompt)
            except MilvusException as e:
                if "vector type must be the same" in str(e):
                    return "There was an issue with the query format. Please try asking a more detailed question."
                else:
                    return f"An error occurred: {e}"

          
            # Append assistant's response to messages and display feedback buttons immediately
            st.session_state['messages'].append({
                "role": "assistant",
                "content": {
                    "response": response
                    
                }
            })
            
            # Display like/dislike buttons for the response
            # display_rating_buttons(len(st.session_state['messages']) - 1)

        response_time = time.time() - start_time
        st.session_state['total_response_time'] += response_time
        st.session_state['num_responses'] += 1

    # Sidebar for Metrics and Reset
    st.sidebar.title("Metric Summary")
    with st.sidebar.expander("Overall Metrics"):
        st.write(f"Total Questions: {st.session_state['num_questions']}")
        st.write(f"Correct Answers: {st.session_state['num_correct_answers']}")
        st.write(f"Incorrect Answers: {st.session_state['num_incorrect_answers']}")

        # Always display the confusion matrix, even if empty
        if st.session_state["y_true"] and st.session_state["y_pred"]:
            cm = confusion_matrix(st.session_state["y_true"], st.session_state["y_pred"])
        else:
            # Create an empty confusion matrix (2x2) if no data is available
            cm = np.array([[0, 0], [0, 0]])

        st.write("Confusion Matrix:")
        st.table(pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"]))
        
        # Display accuracy only if there are predictions
        if st.session_state["y_true"] and st.session_state["y_pred"]:
            accuracy = accuracy_score(st.session_state["y_true"], st.session_state["y_pred"])
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
        else:
            st.write("Accuracy: N/A")

    if st.sidebar.button("Reset Metrics"):
        reset_metrics()
        st.sidebar.success("Metrics have been reset.")

    # Handle user input in chat
    if prompt := st.chat_input("Message Team2 academic chatbot"):
        process_input(prompt)

    # Display chat messages
    for index, message in enumerate(st.session_state.get('messages', [])):
        if message['role'] == 'user':
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            response_content = message['content'].get("response", "")
            st.markdown(response_content)
            # Show like/dislike buttons for each assistant response
            display_rating_buttons(index)

else:
    st.warning("API key is required to proceed.")
