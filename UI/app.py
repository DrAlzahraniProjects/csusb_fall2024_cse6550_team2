import streamlit as st
import time

# Function for chatbot responses
def chatbot_response(user_input):
    responses = {
        'hi': 'Hello! How can I assist you with your academic needs?',
        'bye': 'Goodbye! Feel free to ask if you need more help.',
        'what can you do': 'I can help you with academic advising, research topics, and more!',
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
            <span class="rating-icon" title="Copy">📋</span>
        </div>
    """, unsafe_allow_html=True)

# Custom CSS for styling messages, and logo
primary_color = st.get_option("theme.primaryColor")
background_color = st.get_option("theme.backgroundColor")
text_color = st.get_option("theme.textColor")

st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{
        padding-top: 0;
    }}
    .stTextInput input {{
        background-color: {background_color} !important; /* Dynamic Background */
        color: {text_color} !important; /* Dynamic Text Color */
        border: 1px solid {primary_color} !important; /* Dynamic Border Color */
        border-radius: 5px;
        padding: 15px;
        font-size: 1rem;
        width: 100%;
    }}
    .chat-message {{
        margin: 15px 0;
        padding: 15px;
    }}
    .chat-message-user {{
        text-align: right;
        background-color: #3C3C3C; /* You may also want to make this dynamic */
        color: #FFFFFF;
        border: 1px solid #3C3F41;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        display: inline-block;
        max-width: 70%;
        float: right;
        clear: both;
    }}
    .chat-message-bot {{
        text-align: left;
        background-color: #2B2B2B; /* You may also want to make this dynamic */
        color: #FFFFFF;
        border: 1px solid #3C3F41;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        display: inline-block;
        max-width: 70%;
        float: left;
        clear: both;
    }}
    
    .fixed-logo-text {{
        font-size: 24px;
        color: {text_color}; /* Dynamic Text Color */
        font-weight: bold;
        margin: -60px;
    }}

    .rating-buttons {{
        text-align: left;
        margin: 5px 0;
        padding: 0;
        display: flex;
        gap: 8px;
    }}
    .rating-icon {{
        cursor: pointer;
        font-size: 1rem;
        padding: 1px;
        border-radius: 4px;
        display: inline-block;
        position: relative;
    }}
    .rating-icon:hover {{
        background-color: #444444;
    }}
    </style>
    """, unsafe_allow_html=True)

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
        <div class="fixed-logo-text">Academic Assistant</div>
    """, unsafe_allow_html=True)

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Display conversation history
st.subheader("Conversation History")
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
