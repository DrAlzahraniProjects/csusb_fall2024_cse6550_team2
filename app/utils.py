import streamlit as st
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from collections import defaultdict

# Initialize session state variables
def initialize_session_state():
    if 'y_true' not in st.session_state:
        # Initialize y_true with ground truth values
        ground_truth_data = [
            # Correct questions and answers with y_true = 1
            {"question": "Information about the School of Computer Science & Engineering programs.",
             "answer": "The School offers several programs, including BS and MS in Computer Science, BS in Computer Engineering, BS in Bioinformatics, and BA in Computer Systems.",
             "y_true": 1},
            {"question": "What accreditation does the Bachelor of Science in Computer Science program have?",
             "answer": "The Bachelor of Science in Computer Science program is accredited by the Computing Accreditation Commission of ABET, and the Bachelor of Science in Computer Engineering is accredited by the Engineering Accreditation Commission of ABET.",
             "y_true": 1},
            {"question": "How does the School of Computer Science & Engineering engage with industry for internships and employment?",
             "answer": "The school has strong ties with Southern California employers, offering internships and long-term employment opportunities. It also has an Industry Advisory Board to stay updated on industry trends.",
             "y_true": 1},
            {"question": "Who is the current Chair of the School of Computer Science & Engineering, and how can they be contacted?",
             "answer": "Dr. Khalil Dajani is the Chair/Director. Contact: Office JB 307B, Phone (909) 537-5326, Email: khalil.dajani@csusb.edu.",
             "y_true": 1},
            {"question": "How can students access remote labs in the School of Computer Science & Engineering?",
             "answer": "Remote lab access is supported with instructions like CSE-Jump, Horizon VDI, SSH Remote Access, and WinSCP File Transfer.",
             "y_true": 1},
            {"question": "What is the CSE Club, and what resources does it provide to students?",
             "answer": "The CSE Club, along with resources such as WiCSE and various support seminars, provides networking, support, and career resources for students.",
             "y_true": 1},
            {"question": "What pathways are available in the Certificate in Computer Science for K-12 Educators program?",
             "answer": "The CS Certificate for K-12 Educators offers structured pathways and resources specific to educating young students in computer science basics.",
             "y_true": 1},
            {"question": "How can alumni of the School of Computer Science & Engineering stay involved with the school?",
             "answer": "Alumni can remain involved through industry connections, participating in the advisory board, and engaging in campus events.",
             "y_true": 1},
            {"question": "What resources and support services are available for students at the School of Computer Science & Engineering?",
             "answer": "Resources include academic advising, internship programs, computer labs with remote access, and scholarship opportunities such as the ExCELS Scholarship.",
             "y_true": 1},
            {"question": "What opportunities are available for undergraduate students in the School of Computer Science & Engineering to participate in research and internships?",
             "answer": "There are extensive opportunities through Southern California industry partnerships and the Industry Advisory Board, which help students gain relevant work experience and stay updated with industry needs.",
             "y_true": 1},

            # Incorrect questions with no answers and y_true = 0
            {"question": "What personal contact information is available for all faculty members in the Computer Science department?",
             "answer": "This information is not available due to privacy policies.",
             "y_true": 0},
            {"question": "What specific job positions do alumni hold at top tech companies?",
             "answer": "Detailed information on specific job roles of alumni is typically confidential or not publicly disclosed.",
             "y_true": 0},
            {"question": "How much funding is allocated annually to each research project in the school?",
             "answer": "Funding details for individual projects are generally not disclosed publicly.",
             "y_true": 0},
            {"question": "What is the tuition fee for international students in the School of Computer Science & Engineering?",
             "answer": "Specific tuition fees would typically be available on the CSUSB admissions or finance pages, not within the CSE program documentation.",
             "y_true": 0},
            {"question": "How does the School of Computer Science & Engineering compare to other schools nationally in terms of ranking?",
             "answer": "Comparative rankings are generally found through external ranking organizations rather than internal school documents.",
             "y_true": 0},
            {"question": "What are the housing options available near the School of Computer Science & Engineering?",
             "answer": "Housing information is not specific to the department and would be handled by campus housing services.",
             "y_true": 0},
            {"question": "Can you explain the prerequisites for courses offered at other universities?",
             "answer": "Course prerequisites at other universities would not be detailed in CSUSB’s CSE documentation.",
             "y_true": 0},
            {"question": "What are the exact passing percentages for each class in the Computer Science program?",
             "answer": "Specific passing percentages are typically not disclosed in program details, as grading is dependent on individual course assessments.",
             "y_true": 0},
            {"question": "Can you list personal testimonials from every alumni about their experiences in the program?",
             "answer": "Personal alumni testimonials may be selectively shared but are not comprehensively available in official program information.",
             "y_true": 0},
            {"question": "Can you give a detailed syllabus for every course offered by the School of Computer Science & Engineering?",
             "answer": "Detailed syllabi are generally accessible to enrolled students or via request, not in general program information.",
             "y_true": 0}
        ]
        
        # Populate y_true in session state
        st.session_state["y_true"] = [item["y_true"] for item in ground_truth_data]

    if 'y_pred' not in st.session_state:
        st.session_state["y_pred"] = []

    if 'num_questions' not in st.session_state:
        st.session_state['num_questions'] = 0

    if 'num_correct_answers' not in st.session_state:
        st.session_state['num_correct_answers'] = 0

    if 'num_incorrect_answers' not in st.session_state:
        st.session_state['num_incorrect_answers'] = 0

    if 'user_engagement' not in st.session_state:
        st.session_state['user_engagement'] = {'likes': 0, 'dislikes': 0}

    if 'num_responses' not in st.session_state:
        st.session_state['num_responses'] = 0

    if 'rated_responses' not in st.session_state:
        st.session_state['rated_responses'] = {}

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    if 'total_response_time' not in st.session_state:
        st.session_state['total_response_time'] = 0
     # Initialize variables related to the title animation
    if 'input_given' not in st.session_state:
        st.session_state['input_given'] = False

    if 'title_animated' not in st.session_state:
        st.session_state['title_animated'] = False

    if 'title_placeholder' not in st.session_state:
        st.session_state['title_placeholder'] = st.empty()
    for index, message in enumerate(st.session_state['messages']):
        feedback_key = f"feedback_{index}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None

def typing_title_animation(title, delay=0.1):
    """Animates typing effect for a given title."""
    title_placeholder = st.empty()
    animated_title = ""
    for char in title:
        animated_title += char
        title_placeholder.markdown(f"<h2 style='text-align: center;'>{animated_title}</h2>", unsafe_allow_html=True)
        time.sleep(delay)
    return title_placeholder


def reset_metrics():
    """Resets all tracked metrics in session state."""
    # Reset engagement and response metrics
    # st.session_state['num_questions'] = 0
    st.session_state['num_correct_answers'] = 0
    st.session_state['num_incorrect_answers'] = 0
    st.session_state['user_engagement'] = {'likes': 0, 'dislikes': 0}
    st.session_state['rated_responses'] = {i: None for i in range(len(st.session_state.get('messages', [])))}
    # st.session_state["y_true"] = []
    st.session_state["y_pred"] = []
    st.session_state['total_response_time'] = 0
    
   
    # Clear confusion matrix and performance metrics
    st.session_state["confusion_matrix_placeholder"].empty()
    st.session_state["accuracy_placeholder"].empty()
    st.session_state["precision_placeholder"].empty()
    st.session_state["recall_placeholder"].empty()
    st.session_state["sensitivity_placeholder"].empty()
    st.session_state["specificity_placeholder"].empty()
    feedback_keys = [key for key in st.session_state.keys() if key.startswith("feedback_")]
    for key in feedback_keys:
        del st.session_state[key]
    # Reset feedback keys
    # reset_feedback_keys()
    del st.session_state["reset_button_clicked"]
    # # for debuging
    # print("Rated Responses after reset:", st.session_state['rated_responses'])
    # print("Feedback keys after reset:", [key for key in st.session_state.keys() if key.startswith("feedback_")])

    update_metrics()
    

def initialize_metrics_sidebar():
    """Initializes sidebar placeholders for metrics and confusion matrix in an expanded UI."""
    
    st.sidebar.markdown(
        """
        <h2><a href="https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team2?tab=readme-ov-file#Software-Quality-Assurance" target="_blank" style="text-decoration: none; color: inherit;">Confusion Matrix</a></h2>
        """, 
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        # Confusion Matrix and Performance Metrics
        # st.sidebar.write("Confusion Matrix:")
        
        st.session_state["sensitivity_placeholder"] = st.empty()
        st.session_state["specificity_placeholder"] = st.empty()
        st.session_state["confusion_matrix_placeholder"] = st.empty()
        st.session_state["accuracy_placeholder"] = st.empty()
        st.session_state["precision_placeholder"] = st.empty()
        st.session_state["recall_placeholder"] = st.empty()
    
    # Initial update to display zeroed or default metrics
    update_metrics()

def update_metrics():
    st.markdown(
        """
        <style>
        .metric-box {
            background-color: #f0f8ff;  /* Light gray background */
            border: 1px solid #d0e7ff;  /* Border color */
            border-radius: 8px;  /* Rounded corners */
            padding: 10px;  /* Inner spacing */
            margin-bottom: 10px;  /* Space between boxes */
            font-family: Arial, sans-serif;  /* Font style */
            font-size: 14px;  /* Text size */
            color: skyblue;  /* Text color */
            font-weight: bold;  /* Bold text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .performance-box {
            background-color: #f4f4f4;  /* Light gray background */
            border: 1px solid #ccc;  /* Border color */
            border-radius: 8px;  /* Rounded corners */
            padding: 10px;  /* Inner spacing */
            margin-bottom: 10px;  /* Space between boxes */
            font-family: Arial, sans-serif;  /* Font style */
            font-size: 14px;  /* Text size */
            color: gray;  /* Text color */
            font-weight: bold;  /* Bold text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    .curved-table {
        border-collapse: separate; /* Allows border-radius to take effect */
        border-spacing: 0; /* Ensures there’s no unwanted spacing */
        border: 1px solid #ccc; /* Adds a border around the table */
        border-radius: 12px; /* Rounds the table's corners */
        overflow: hidden; /* Ensures rounded corners display properly */
    }
    .curved-table th, .curved-table td {
        border: 1px solid #ccc; /* Adds borders to cells */
        padding: 8px; /* Adds padding for readability */
        text-align: center; /* Centers the content */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    # print('start update')
    """Updates metrics such as confusion matrix, accuracy, specificity, precision, recall, and sensitivity in the sidebar."""
    if st.session_state["y_true"] and st.session_state["y_pred"]:
        adjusted_y_true = st.session_state["y_true"][:len(st.session_state["y_pred"])]
        cm = confusion_matrix(adjusted_y_true, st.session_state["y_pred"], labels=[1, 0])
        accuracy = accuracy_score(adjusted_y_true, st.session_state["y_pred"])
        precision = precision_score(adjusted_y_true, st.session_state["y_pred"], zero_division=0)
        recall = recall_score(adjusted_y_true, st.session_state["y_pred"], zero_division=0)

        # Ensure confusion matrix is always 2x2
        if cm.size == 1:
            cm = np.array([[cm[0, 0], 0], [0, 0]]) if adjusted_y_true[0] == 0 else np.array([[0, 0], [0, cm[0, 0]]])
        elif cm.shape == (1, 2):
            cm = np.vstack((cm, [0, 0])) if adjusted_y_true[0] == 0 else np.vstack(([0, 0], cm))
        elif cm.shape == (2, 1):
            cm = np.hstack((cm, [[0], [0]]))

        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
    
        # Calculate specificity and sensitivity
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        sensitivity = recall  # Sensitivity is equivalent to recall in binary classification
        # st.sidebar.markdown(
        #     """
        #     <h2><a href="https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team2?tab=readme-ov-file#Software-Quality-Assurance" target="_blank" style="text-decoration: none; color: inherit;">Confusion Matrix</a></h2>
        #     """, 
        #     unsafe_allow_html=True
        # )
        st.session_state["sensitivity_placeholder"].markdown(
            f"""
            <div class="metric-box">Sensitivity: {sensitivity * 100:.2f}%</div>
            """,
            unsafe_allow_html=True
        )
        st.session_state["specificity_placeholder"].markdown(
            f"""
            <div class="metric-box">Specificity: {specificity * 100:.2f}%</div>
            """,
            unsafe_allow_html=True
        )
        
        # Update confusion matrix and metrics in the sidebar
        # st.session_state["confusion_matrix_placeholder"].table(
        #     pd.DataFrame(cm, columns=["Pred. Ans", "Pred. Unans"], index=["Actual Ans", "Actual Unans"])
        # )
        st.session_state["confusion_matrix_placeholder"].markdown(
            pd.DataFrame(cm, columns=["Predicted +", "Predicted -"], index=["Actual +", "Actual -"])
            .to_html(classes="curved-table", escape=False),
            unsafe_allow_html=True
        )
        # Accuracy, Precision, and Recall
        st.session_state["accuracy_placeholder"].markdown(
            f"""
            <div class="performance-box">Accuracy: {accuracy * 100:.2f}%</div>
            """,
            unsafe_allow_html=True
        )
        # st.session_state["accuracy_placeholder"].write(f"Accuracy: {accuracy * 100:.2f}%")
        st.session_state["precision_placeholder"].markdown(
            f"""
            <div class="performance-box">Precision: {precision * 100:.2f}%</div>
            """,
            unsafe_allow_html=True
        )
        st.session_state["recall_placeholder"].markdown(
            f"""
            <div class="performance-box">F1 Score: {recall * 100:.2f}%</div>
            """,
            unsafe_allow_html=True
        )
        # print("Updates metrics:data exist")

        # st.session_state["precision_placeholder"].write(f"Precision: {precision * 100:.2f}%")
        # st.session_state["recall_placeholder"].write(f"Recall: {recall * 100:.2f}%")

    else:
        st.session_state["sensitivity_placeholder"].markdown("<div class='metric-box'>Sensitivity: 0.0</div>", unsafe_allow_html=True)
        st.session_state["specificity_placeholder"].markdown("<div class='metric-box'>Specificity: 0.0</div>", unsafe_allow_html=True)
        # st.session_state["confusion_matrix_placeholder"].table(
        #     pd.DataFrame( columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])
        # )
        # Render the table
        cm_values = [
            [f"{0} (TP)", f"{0} (FN)"],  # First row: Pred. Ans, Pred. Unans
            [f"{0} (FP)", f"{0} (TN)"]   # Second row: Actual Ans, Actual Unans
        ]
        conf_matrix_df = pd.DataFrame(
            cm_values, 
            columns=["Predicted +", "Predicted -"], 
            index=["Actual +", "Actual -"]
        )
        st.session_state["confusion_matrix_placeholder"].markdown(
            conf_matrix_df.to_html(classes="curved-table", escape=False),
            unsafe_allow_html=True
        )
        
        # st.session_state["confusion_matrix_placeholder"].markdown(
        #     pd.DataFrame(columns=["Pred. Unans", "Pred. Ans"], index=["Actual Unans", "Actual ans"])
        #     .to_html(classes="curved-stable", escape=False),
        #     unsafe_allow_html=True
        # )
        #st.session_state["confusion_matrix_placeholder"].write("Confusion Matrix: No data available.")
        st.session_state["accuracy_placeholder"].markdown("<div class='performance-box'>Accuracy: 0.0</div>", unsafe_allow_html=True)
        # st.session_state["accuracy_placeholder"].write("Accuracy: N/A")
        st.session_state["precision_placeholder"].markdown("<div class='performance-box'>Precision: 0.0</div>", unsafe_allow_html=True)
        st.session_state["recall_placeholder"].markdown("<div class='performance-box'>F1 Score: 0.0</div>", unsafe_allow_html=True)
        # st.session_state["precision_placeholder"].write("Precision: N/A")
        # st.session_state["recall_placeholder"].write("Recall: N/A")
    #     print("Updates metrics:empty")

    # print("Updates metrics:")

def handle_feedback(index):
    """Handles user feedback for a specific response."""
    feedback_value = st.session_state.get(f"feedback_{index}")
    print(f"Feedback received for response at index {index},{feedback_value}")
    if feedback_value is not None:  # Ensure feedback is not None
        previous_rating = st.session_state['rated_responses'].get(index)
        
        if feedback_value == 1:  # Thumbs up
            update_likes(index, previous_rating)
        elif feedback_value == 0:  # Thumbs downhe
            update_dislikes(index, previous_rating)


def update_likes(index, previous_rating):
    """Updates metrics when a response is liked."""
    #for debuging
    # print("UPDATE LIKE")
    # print(previous_rating)
    if previous_rating != 'liked':  # Avoid duplicate likes
        if previous_rating == 'disliked':
            st.session_state['num_incorrect_answers'] -= 1
            st.session_state["y_pred"].remove(0)  # Remove dislike

        st.session_state['num_correct_answers'] += 1
        st.session_state['user_engagement']['likes'] += 1
        st.session_state['rated_responses'][index] = 'liked'
        st.session_state["y_pred"].append(1)
        update_metrics()

def update_dislikes(index, previous_rating):
    """Updates metrics when a response is disliked."""
    #for debuging
    # print("UPDATE DISLIKE")
    print(previous_rating)
    if previous_rating != 'disliked':  # Avoid duplicate dislikes
        if previous_rating == 'liked':
            st.session_state['num_correct_answers'] -= 1
            st.session_state["y_pred"].remove(1)  # Remove like

        st.session_state['num_incorrect_answers'] += 1
        st.session_state['user_engagement']['dislikes'] += 1
        st.session_state['rated_responses'][index] = 'disliked'
        st.session_state["y_pred"].append(0)
        update_metrics()

# Rate-limiting setup
REQUEST_LOG = defaultdict(list)  # Tracks requests per IP
MAX_REQUESTS = 15     # Max requests allowed
WINDOW_SECONDS = 180  # Time window in seconds

def is_rate_limited(ip, action_type="general"):
    """Check if an IP is exceeding the request limit based on action type."""
    now = time.time()

    # Only track rate-limiting for general actions (like querying the model)
    if action_type == "general":
        REQUEST_LOG[ip] = [timestamp for timestamp in REQUEST_LOG[ip] if now - timestamp < WINDOW_SECONDS]
        if len(REQUEST_LOG[ip]) >= MAX_REQUESTS:
            return True
        REQUEST_LOG[ip].append(now)
    
    return False  # No rate-limiting for non-general actions
