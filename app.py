import streamlit as st
import os
import subprocess
import time
from Inference import *

corpus_source = [
    "https://www.csusb.edu/cse",
    "https://catalog.csusb.edu/"
]
st.set_page_config(page_title = "Academic Chatbot - Team2")

def main():

    # Initialize session state for input tracking
    if 'input_given' not in st.session_state:
        st.session_state['input_given'] = False  # Track if input has been given

    if 'title_animated' not in st.session_state:
        st.session_state['title_animated'] = False  # Track if the title has been animated

    if 'title_placeholder' not in st.session_state:
        st.session_state['title_placeholder'] = st.empty()  # Initialize placeholder for the title

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
        placeholder = st.empty()  # Create a placeholder to update dynamically
        words = title.split()
        full_text = ""
        for word in words:
            full_text += word + " "
            # Center and style the title
            placeholder.markdown(
                f"<h1 style='text-align: center; font-size: 36px; font-weight: bold; color: #333;'>{full_text.strip()}</h1>",
                unsafe_allow_html=True
            )
            time.sleep(delay)
        return placeholder

    def update_likes(index):
        previous_rating = st.session_state['rated_responses'].get(index)
        if previous_rating != 'liked':
            # Update from disliked or neutral to liked
            if previous_rating == 'disliked':
                st.session_state['num_incorrect_answers'] -= 1
                st.session_state['user_engagement']['dislikes'] -= 1
            st.session_state['num_correct_answers'] += 1
            st.session_state['user_engagement']['likes'] += 1
            st.session_state['rated_responses'][index] = 'liked'

    def update_dislikes(index):
        previous_rating = st.session_state['rated_responses'].get(index)
        if previous_rating != 'disliked':
            # Update from liked or neutral to disliked
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


        # Recalculate accuracy rate
        if st.session_state['num_questions'] > 0:
            accuracy_rate = (st.session_state['num_correct_answers'] / st.session_state['num_questions']) * 100
            st.session_state['accuracy_rate'] = accuracy_rate


    # Apply the external CSS file
    with open("./style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Sidebar for chat history and statistics
    # Sidebar for chat history and statistics
    st.sidebar.title("Metric Summary")

    # Number of questions
    with st.sidebar.expander("Number of questions"):
        st.write(f"{st.session_state['num_questions']}")

    # Number of correct answers
    with st.sidebar.expander("Number of correct answers"):
        st.write(f"{st.session_state['num_correct_answers']}")

    # Number of incorrect answers
    with st.sidebar.expander("Number of incorrect answers"):
        st.write(f"{st.session_state['num_incorrect_answers']}")

    # User engagement metrics (likes/dislikes)
    with st.sidebar.expander("User engagement metrics"):
        st.write(f"👍 Likes: {st.session_state['user_engagement']['likes']}")
        st.write(f"👎 Dislikes: {st.session_state['user_engagement']['dislikes']}")

    # Response time analysis
    with st.sidebar.expander("Response time analysis"):
        if st.session_state['num_responses'] > 0:
            avg_response_time = st.session_state['total_response_time'] / st.session_state['num_responses']
            st.write(f"Average Response Time: {avg_response_time:.2f} seconds")
        else:
            st.write("No responses yet.")

    # Accuracy rate
    with st.sidebar.expander("Accuracy rate"):
        if st.session_state['num_questions'] > 0:
            accuracy_rate = (st.session_state['num_correct_answers'] / st.session_state['num_questions']) * 100
            st.write(f"Accuracy Rate: {accuracy_rate:.2f}%")
        else:
            st.write("No questions answered yet.")
        
    # Needs to be implemented
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
        if not st.session_state['input_given'] and not st.session_state['title_animated']:
    # Only animate the title if no input has been given and it hasn't been animated yet
            st.session_state['title_placeholder'] = typing_title_animation("Academic Advisor Chatbot", delay=0.3)
            st.session_state['title_animated'] = True
    else:
        # Display the fixed title at the top left with a logo if input is given
        st.markdown(f"""<div class="chat-title">Academic Advisor Chatbot</div>""", unsafe_allow_html=True)

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
        with st.spinner("Initializing, Please Wait..."):
            vector_store = initialize_milvus()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [] 

    # Function to process user input and generate bot response
    def process_input(prompt):
        st.session_state['num_questions'] += 1
        st.session_state['messages'].append({"role": "user", "content": prompt})

        start_time = time.time()

        with st.spinner('Generating Response...'):
            response,sources, images = invoke_llm_for_response(prompt)
            
        print(f"Response: {response}, Source URL: {sources}, Image: {images}")  # Ensure this returns a tuple
        #    # Display sources only if they exist
        # if sources:
        #     st.subheader("Sources:")
        #     st.write(sources)
        # else:
        #     st.write("No sources found for this response.")

        # # Display images only if they exist
        # if images:
        #     st.subheader("Associated Images:")
        #     for image_path in images:
        #         st.image(image_path, caption=os.path.basename(image_path))
        # else:
        #     st.write("No associated images found for this response.")
        response_time = time.time() - start_time
        st.session_state['total_response_time'] += response_time
        st.session_state['num_responses'] += 1

        st.session_state['messages'].append({
            "role": "assistant",
            "content": {
                "response": response,
                "source": sources
            }
        })



    # Handle user input
    if prompt := st.chat_input("Message Team2 academic chatbot"):
        process_input(prompt)

    for index, message in enumerate(st.session_state.get('messages', [])):
        if message['role'] == 'user':
            # Display user message with right-aligned CSS styling
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            # Display assistant message with left-aligned CSS styling
            if isinstance(message['content'], dict):
                response_content = message['content'].get("response", "")
                source_url = message['content'].get("source", "Unknown Source")
            else:
                # Fallback if content is a tuple or another type
                response_content, source_url = message['content'] if isinstance(message['content'], tuple) else (message['content'], "Unknown Source")
            
            # Display assistant response content with styling
            col1, col2 = st.columns(2)
            with col1:
              st.markdown(response_content)
            
            # Show source if available
            # if source_url and source_url != "Unknown Source":
            #     st.markdown(f"<div class='assistant-message'><strong>Source</strong>: <a href='{source_url}' target='_blank'>{source_url}</a></div>", unsafe_allow_html=True)
            # else:
            #     st.markdown(f"<div class='assistant-message'><strong>Source</strong>: Unknown Source</div>", unsafe_allow_html=True)
    
            # Display rating buttons below the assistant message
            display_rating_buttons(index)
    


                    

if __name__ == "__main__":
    main()