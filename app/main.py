# Description: This file contains the main Streamlit application code for the Academic Chatbot project.
import streamlit as st
import os
import time
from pymilvus import MilvusException
import backend

# Set page configuration
st.set_page_config(page_title="Academic Chatbot - Team2")

# Prompt the user to enter their API key if not provided yet
if "api_key" not in st.session_state:
    api_key = st.text_input("Please enter your Mistral API key:", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
        os.environ["API_KEY"] = api_key  # Set the API key in the environment
else:
    api_key = st.session_state["api_key"]

# Only proceed if API key is set
if "API_KEY" in os.environ:
    # Initialize session state for input tracking
    if 'input_given' not in st.session_state:
        st.session_state['input_given'] = False

    if 'title_animated' not in st.session_state:
        st.session_state['title_animated'] = False

    if 'title_placeholder' not in st.session_state:
        st.session_state['title_placeholder'] = st.empty()

    # Initialize metrics in session state
    if 'num_questions' not in st.session_state:
        st.session_state['num_questions'] = 0

    if 'num_correct_answers' not in st.session_state:
        st.session_state['num_correct_answers'] = 0

    if 'num_incorrect_answers' not in st.session_state:
        st.session_state['num_incorrect_answers'] = 0

    if 'total_response_time' not in st.session_state:
        st.session_state['total_response_time'] = 0.0

    if 'num_responses' not in st.session_state:
        st.session_state['num_responses'] = 0

    if 'user_engagement' not in st.session_state:
        st.session_state['user_engagement'] = {'likes': 0, 'dislikes': 0}

    if 'rated_responses' not in st.session_state or not isinstance(st.session_state['rated_responses'], dict):
        st.session_state['rated_responses'] = {}

    # CSS styling
    with open("./style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Function for animated typing title
    def typing_title_animation(title, delay=0.3):
        placeholder = st.empty()
        words = title.split()
        full_text = ""
        for word in words:
            full_text += word + " "
            placeholder.markdown(
                f"<h1 style='text-align: center; font-size: 36px; font-weight: bold; color: #333;'>{full_text.strip()}</h1>",
                unsafe_allow_html=True
            )
            time.sleep(delay)
        return placeholder

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
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍", key=f"like_button_{index}", on_click=update_likes, args=(index,))
        with col2:
            st.button("👎", key=f"dislike_button_{index}", on_click=update_dislikes, args=(index,))

        if st.session_state['num_questions'] > 0:
            accuracy_rate = (st.session_state['num_correct_answers'] / st.session_state['num_questions']) * 100
            st.session_state['accuracy_rate'] = accuracy_rate

    # Sidebar metrics
    st.sidebar.title("Metric Summary")
    with st.sidebar.expander("Number of questions"):
        st.write(f"{st.session_state['num_questions']}")
    with st.sidebar.expander("Number of correct answers"):
        st.write(f"{st.session_state['num_correct_answers']}")
    with st.sidebar.expander("Number of incorrect answers"):
        st.write(f"{st.session_state['num_incorrect_answers']}")
    with st.sidebar.expander("User engagement metrics"):
        st.write(f"👍 Likes: {st.session_state['user_engagement']['likes']}")
        st.write(f"👎 Dislikes: {st.session_state['user_engagement']['dislikes']}")
    with st.sidebar.expander("Response time analysis"):
        if st.session_state['num_responses'] > 0:
            avg_response_time = st.session_state['total_response_time'] / st.session_state['num_responses']
            st.write(f"Average Response Time: {avg_response_time:.2f} seconds")
        else:
            st.write("No responses yet.")

    with st.sidebar.expander("Accuracy rate"):
        if st.session_state['num_questions'] > 0:
            accuracy_rate = (st.session_state['num_correct_answers'] / st.session_state['num_questions']) * 100
            st.write(f"Accuracy Rate: {accuracy_rate:.2f}%")
        else:
            st.write("No questions answered yet.")

    # Placeholder for the animated title
    if 'title_placeholder' not in st.session_state:
        st.session_state['title_placeholder'] = st.empty()

    if not st.session_state['input_given'] and not st.session_state['title_animated']:
        st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
        st.session_state['title_animated'] = True
    else:
        st.markdown(f"""<div class="chat-title">Academic Advisor Chatbot</div>""", unsafe_allow_html=True)

    # Initialize Milvus if not already done
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
        with st.spinner("Initializing, Please Wait..."):
            backend.initialize_milvus()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Function to process user input and generate bot response
    def process_input(prompt):
        st.session_state['num_questions'] += 1
        st.session_state['messages'].append({"role": "user", "content": prompt})

        start_time = time.time()
        
        with st.spinner('Generating Response...'):
            try:
                response,sources = backend.invoke_llm_for_response(prompt)
                
            except MilvusException as e:
                if "vector type must be the same" in str(e):
                    return "There was an issue with the query format. Please try asking a more detailed question."
                else:
                    return f"An error occurred: {e}"
        
        response_time = time.time() - start_time
        st.session_state['total_response_time'] += response_time
        st.session_state['num_responses'] += 1

        st.session_state['messages'].append({
            "role": "assistant",
            "content": {
                "response": response,
                "sources": sources
            }
        })

    # Handle user input
    if prompt := st.chat_input("Message Team2 academic chatbot"):
        process_input(prompt)

    for index, message in enumerate(st.session_state.get('messages', [])):
        if message['role'] == 'user':
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            response_content = message['content'].get("response", "")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(response_content)
            display_rating_buttons(index)

else:
    st.warning("API key is required to proceed.")
