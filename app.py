import streamlit as st
import os
import subprocess
from RAG import *
# Changes tab title (Warning: Leave at top)
st.set_page_config(page_title = "Academic Chatbot - Team2")
def main():
    """Main Streamlit app logic."""
    header = st.container()

    def load_css(file_name):
        try:
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"css file '{file_name}' not found.")

    # Load the CSS file
    load_css("assets/style.css")

    # Add custom CSS for buttons and alignment
    st.markdown("""
        <style>
        .assistant-message {
            margin-bottom: 0; /* Remove extra space below the message */
        }
        .feedback-buttons {
            display: inline-flex;  /* Make buttons inline */
            gap: 5px;  /* Reduce gap between buttons */
            margin-top: 5px;  /* Minimize vertical gap */
        }
        button[aria-label="👍 Like"], button[aria-label="👎 Dislike"] {
            background-color: transparent;
            border: none;
            cursor: pointer;
            font-size: 20px;
        }
        button[aria-label="👍 Like"]:hover::after {
            content: 'Like';  /* Display "Like" without emoji on hover */
            font-size: 14px;
            color: #000;
            position: absolute;
            top: 40px; /* Position text below the button */
        }
        button[aria-label="👎 Dislike"]:hover::after {
            content: 'Dislike';  /* Display "Dislike" without emoji on hover */
            font-size: 14px;
            color: #000;
            position: absolute;
            top: 40px; /* Position text below the button */
        }
        </style>
    """, unsafe_allow_html=True)

    header.write("""<div class='chat-title'>Team 2 CSE Academic Advisor Chatbot</div>""", unsafe_allow_html=True)
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # Sidebar for chat history and statistics
    st.sidebar.title("10 Statistics Reports")

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

if __name__ == "__main__":
    # If streamlit instance is running
    if os.environ.get("STREAMLIT_RUNNING") == "1":
        main()
    else:
        os.environ["STREAMLIT_RUNNING"] = "1"  # Set the environment variable to indicate Streamlit is running
		#if multiple processes are being started, you must use Popen followed by run subprocess!
        subprocess.run(["streamlit", "run", __file__, "--server.port=5002", "--server.address=0.0.0.0", "--server.baseUrlPath=/team2"])
        #subprocess.run(["jupyter", "notebook", "--ip=0.0.0.0", "--port=6002", "--no-browser", "--allow-root"])