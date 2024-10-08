import streamlit as st
import time
# Changes tab title (Warning: Leave at top)
st.set_page_config(page_title = "Academic Chatbot - Team2")

# CSS styling
with open("assets/style.css") as f:
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