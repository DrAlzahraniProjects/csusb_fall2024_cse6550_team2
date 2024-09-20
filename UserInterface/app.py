import streamlit as st

# st.title("Hello World! - Team 2")

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

# Sidebar for additional options (e.g., navigation or tips)

st.sidebar.title("Academic Chatbot")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- Ask about academic advice")
st.sidebar.markdown("- Get help with research topics")
st.sidebar.markdown("- Explore course recommendations")
st.sidebar.markdown("---")
st.sidebar.markdown("Select options from the left to explore more features.")

# Main Container for chat
st.title("Academic Advisor Chatbot")

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Layout: Left for conversation history, right for input and buttons
left_column, right_column = st.columns([1, 3])

with left_column:
    # Display conversation history
    st.subheader("Conversation History")
    for message in st.session_state['conversation']:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

with right_column:
    st.subheader("Chatbot")
    
    # Input for user message
    user_input = st.text_input("You: ", key="user_input", placeholder="Ask me anything academic...")

    # Buttons for feedback
    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 1])
    
    with feedback_col1:
        like_button = st.button("👍 Like")
    
    with feedback_col2:
        dislike_button = st.button("👎 Dislike")
    
    with feedback_col3:
        copy_button = st.button("📋 Copy")

    # Process user input and display bot response
    if user_input:
        # Append user input to conversation
        st.session_state['conversation'].append({"role": "user", "content": user_input})
        
        # Generate and display bot response
        bot_reply = chatbot_response(user_input)
        st.session_state['conversation'].append({"role": "bot", "content": bot_reply})
        
        # Rerun to update the conversation and clear input
        st.experimental_rerun()

    # Handle button interactions (optional)
    if like_button:
        st.success("You liked the response!")
        
    if dislike_button:
        st.error("You disliked the response!")
        
    if copy_button:
        st.info("Response copied to clipboard!")

