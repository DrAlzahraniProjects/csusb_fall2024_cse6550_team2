# Imports
from initialize_milvus import initialize_milvus_insert_data
from utils import initialize_metrics_sidebar, initialize_session_state, is_rate_limited, update_metrics, reset_metrics, typing_title_animation, update_likes, update_dislikes, handle_feedback
import streamlit as st
import os
import time
from pymilvus import MilvusException
from backend import *
import httpx

# Page configuration
st.set_page_config(page_title="Academic Chatbot - Team2")

# Load CSS styling
with open("./style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Get client IP for rate limiting
client_ip = st.session_state.get("client_ip", "unknown") 

if is_rate_limited(client_ip,action_type="general"):
    st.warning("Too many requests! Please wait 2 minutes before trying again.")
else:
    # Proceed only if API key is set
    if "API_KEY" in os.environ:
        initialize_metrics_sidebar()
        # title_placeholder = st.empty()  # Placeholder for title
        # Typing animation for title
        if not st.session_state['input_given'] and not st.session_state['title_animated']:
            st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.1)
            st.session_state['title_animated'] = True
        else:
            st.markdown(f"""<h2 style='text-align: center;'>Academic Advisor Chatbot</h2>""", unsafe_allow_html=True)

        # Ensure `initialize_and_scrape()` runs only once
        if 'milvus_initialized' not in st.session_state:
            st.session_state['milvus_initialized'] = False  # Default state

            # Placeholder for the dynamic loader
            spinner_placeholder = st.empty()
            initialization_time = 60 # Estimated initialization time in seconds

            # Display spinner and dynamic timer
            with st.spinner("Initializing Milvus..."):
                for remaining_time in range(initialization_time, 0, -1):
                    # Calculate minutes and seconds
                    minutes, seconds = divmod(remaining_time, 60)

                    # Update the timer in the UI
                    spinner_placeholder.markdown(
                        f"<h4 style='text-align: center;'>Please wait for {minutes} minute(s) {seconds} second(s)</h4>",
                        unsafe_allow_html=True
                    )

                    # Run Milvus initialization in the first second
                    if remaining_time == initialization_time:
                        
                        initialize_milvus_insert_data()

                    # Exit the loop if initialization completes early
                    if st.session_state.get('milvus_initialized', False):
                        break

                    time.sleep(0.2)  # Wait for 1 second

            # Clear the spinner and show success or error message
            spinner_placeholder.empty()
        # st.session_state['milvus_initialized'] = True
        # spinner_placeholder.empty()  # Clear the spinner
        # Function to process user input and generate bot response
        def process_input(prompt):
            st.session_state['num_questions'] += 1
            st.session_state['messages'].append({"role": "user", "content": prompt})

            start_time = time.time()
            with st.spinner('Generating Response, Please Wait...'):
                try:
                    response = invoke_llm_for_response(prompt)
                    time.sleep(2) 
                except MilvusException as e:
                    response = "Error: Query format issue. Try a more detailed question." if "vector type must be the same" in str(e) else f"Error: {e}"
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        response = "Rate limit exceeded. Please wait a moment before trying again."
                    else:
                        response = f"An error occurred: {e}"
                except Exception as e:
                    response = f"Unexpected error: {e}"
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
        # Make sure the button's state persists across reruns using session_state
        if "reset_button_clicked" not in st.session_state:
            # reset_metrics()
            st.session_state["reset_button_clicked"] = False  # Initialize state variable

        if st.sidebar.button("Reset Metrics"):
            reset_metrics()  # Reset all metrics and refresh the sidebar with zeroed values
            # reset_feedback()
            st.rerun()
            st.session_state["reset_button_clicked"] = True  # Mark the button as clicked

    else:
        st.warning("API key is required to proceed.")
