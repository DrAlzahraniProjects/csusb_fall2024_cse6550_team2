import streamlit as st

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
    # Append user input to conversation
    st.session_state['conversation'].append({"role": "user", "content": user_input})
    # Generate and append bot response
    bot_reply = chatbot_response(user_input)
    st.session_state['conversation'].append({"role": "bot", "content": bot_reply})
    # Clear the input field after processing
    st.session_state['user_input'] = ''

# Function to display rating buttons for each bot response
def display_rating_buttons(index):
    st.markdown(f"""
        <div class="rating-buttons">
            <span class="rating-icon" title="Like">👍</span>
            <span class="rating-icon" title="Dislike">👎</span>
            <span class="rating-icon" title="Copy">📋</span>
        </div>
    """, unsafe_allow_html=True)

# Custom CSS for styling the input bar and messages
st.markdown("""
    <style>
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #1E1E1E;
        padding: 15px 0;
        border-top: 1px solid #444;
        z-index: 10;
    }
    .stTextInput input {
        background-color: #3C3F41 !important;
        color: #FFFFFF !important;
        border: 1px solid #5A5A5A !important;
        border-radius: 5px;
        padding: 15px;
        font-size: 1rem;
        width: 100%;
    }
    .chat-message {
        margin: 15px 0;
        padding: 15px;
    }
    .chat-message-user {
        text-align: right; /* Align user messages to the right */
        background-color: #3C3C3C; /* Grey background for user messages */
        color: #FFFFFF; /* White text for user messages */
        border: 1px solid #3C3F41;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        display: inline-block;
        max-width: 70%;
        float: right; /* Float user messages to the right */
        clear: both; /* Clear floats */
    }
    .chat-message-bot {
        text-align: left; /* Align bot messages to the left */
        background-color: #2B2B2B;
        color: #FFFFFF; /* White text for bot messages */
        border: 1px solid #3C3F41;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        display: inline-block;
        max-width: 70%;
        float: left; /* Float bot messages to the left */
        clear: both; /* Clear floats */
    }
    .rating-buttons {
        text-align: left; /* Align the rating buttons to the left */
        margin: 5px 0;
        padding: 0;
        display: flex; /* Use flex to position icons horizontally */
        gap: 8px; /* Space between icons */
    }
    .rating-icon {
        cursor: pointer; /* Show pointer cursor for icons */
        font-size: 1rem; /* Smaller icon size */
        padding: 2px;
        border-radius: 4px;
        display: inline-block;
        position: relative;
    }
    .rating-icon:hover {
        background-color: #444444; /* Change background on hover */
    }
    .rating-icon::after {
        content: attr(title); /* Tooltip content */
        display: block;
        position: absolute;
        background: #444444;
        color: #FFFFFF;
        padding: 4px 8px;
        font-size: 0.75rem; /* Smaller font size for tooltip */
        border-radius: 5px;
        visibility: hidden;
        opacity: 0;
        transform: translate(-50%, -20px);
        white-space: nowrap;
        pointer-events: none;
        transition: all 0.2s ease;
        bottom: -30px; /* Position tooltip below the icon */
        left: 50%; /* Center the tooltip */
    }
    .rating-icon:hover::after {
        visibility: visible;
        opacity: 1;
        transform: translate(-50%, -10px); /* Position tooltip slightly above original position */
    }
    .stButton button {
        background-color: #3C3F41 !important;
        color: #FFFFFF !important;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 2px;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for additional options
st.sidebar.title("Academic Chatbot")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- Ask about academic advice")
st.sidebar.markdown("- Get help with research topics")
st.sidebar.markdown("- Explore course recommendations")
st.sidebar.markdown("---")
st.sidebar.markdown("Select options from the left to explore more features.")

# Main Container for chat
st.title("Academic Advisor Chatbot")

# Initialize session state for conversation history and feedback
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = []

# Display conversation history with rating buttons
st.subheader("Conversation History")
for index, message in enumerate(st.session_state['conversation']):
    if message['role'] == 'user':
        st.markdown(f'<div class="chat-message chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-message-bot">{message["content"]}</div>', unsafe_allow_html=True)
        display_rating_buttons(index)

# Input box fixed at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.text_input(
    "You: ",
    key="user_input",
    placeholder="Ask me anything academic...",
    on_change=process_input,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)
