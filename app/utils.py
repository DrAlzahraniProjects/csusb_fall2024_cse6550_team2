import streamlit as st
import time

def typing_title_animation(title, delay=0.1):
    """
    Animates typing effect for a given title.
    Args:
        title (str): The title text to animate.
        delay (float): Delay in seconds between each character (default: 0.1).
    """
    title_placeholder = st.empty()
    animated_title = ""

    # Typing effect loop
    for char in title:
        animated_title += char
        title_placeholder.markdown(f"<h1 style='text-align: center;'>{animated_title}</h1>", unsafe_allow_html=True)
        time.sleep(delay)  # Pause between each character

    return title_placeholder

def reset_metrics():
    """Resets all tracked metrics in session state."""
    st.session_state['num_questions'] = 0
    st.session_state['num_correct_answers'] = 0
    st.session_state['num_incorrect_answers'] = 0
    st.session_state['num_responses'] = 0
    st.session_state['user_engagement'] = {'likes': 0, 'dislikes': 0}
    st.session_state['rated_responses'] = {}
    st.session_state["y_true"] = []
    st.session_state["y_pred"] = []
    st.session_state['total_response_time'] = 0
