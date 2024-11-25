import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

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


def typing_title_animation(title, delay=0.1):
    """Animates typing effect for a given title."""
    title_placeholder = st.empty()
    animated_title = ""
    for char in title:
        animated_title += char
        title_placeholder.markdown(f"<h1 style='text-align: center;'>{animated_title}</h1>", unsafe_allow_html=True)
        time.sleep(delay)
    return title_placeholder

def reset_metrics():
    """Resets all tracked metrics in session state."""
    # Reset engagement and response metrics
    st.session_state['num_questions'] = 0
    st.session_state['num_correct_answers'] = 0
    st.session_state['num_incorrect_answers'] = 0
    st.session_state['user_engagement'] = {'likes': 0, 'dislikes': 0}
    st.session_state['rated_responses'] = {}
    st.session_state["y_true"] = []
    st.session_state["y_pred"] = []
    st.session_state['total_response_time'] = 0 
    
    # Clear confusion matrix and performance metrics
    if "confusion_matrix_placeholder" in st.session_state:
        st.session_state["confusion_matrix_placeholder"].empty()
    if "accuracy_placeholder" in st.session_state:
        st.session_state["accuracy_placeholder"].empty()
    if "precision_placeholder" in st.session_state:
        st.session_state["precision_placeholder"].empty()
    if "recall_placeholder" in st.session_state:
        st.session_state["recall_placeholder"].empty()
    if "sensitivity_placeholder" in st.session_state:
        st.session_state["sensitivity_placeholder"].empty()
    if "specificity_placeholder" in st.session_state:
        st.session_state["specificity_placeholder"].empty()
    
    # Clear engagement metrics placeholders
    if "total_questions_placeholder" in st.session_state:
        st.session_state["total_questions_placeholder"].empty()
    if "correct_answers_placeholder" in st.session_state:
        st.session_state["correct_answers_placeholder"].empty()
    if "incorrect_answers_placeholder" in st.session_state:
        st.session_state["incorrect_answers_placeholder"].empty()
    
    # Optionally, you could call `update_metrics()` here if you want to immediately reset the values to zero in the sidebar
    update_metrics()


def initialize_metrics_sidebar():
    """Initializes sidebar placeholders for metrics and confusion matrix in an expanded UI."""
    
    # Sidebar title with "Confusion Matrix" and a link icon next to it
    st.sidebar.markdown(
        """
        <h2 style='display: inline;'>Confusion Matrix</h2>
        <a href="https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team2/blob/main/README.md" target="_blank" style="text-decoration: none;">
            🔗
        </a>
        """,
        unsafe_allow_html=True
    )
    
    # Initial update to display zeroed or default metrics
    update_metrics()


def update_metrics():
    """Updates metrics such as confusion matrix, accuracy, specificity, precision, recall, and sensitivity in the sidebar."""
     # Function to apply color based on cell content
    def color_cells(val):
        """
        Apply color based on cell content or other logic.
        Args:
            val (str): The value in the cell
        Returns:
            str : The color to apply to the cell
        """
        if "TP" in val or "TN" in val:
            return "background-color: #f1f1f1; color: #444444"
        return "background-color: #f9f9f9; color: #444444"
    
    if st.session_state["y_true"] and st.session_state["y_pred"]:
        adjusted_y_true = st.session_state["y_true"][:len(st.session_state["y_pred"])]
        cm = confusion_matrix(adjusted_y_true, st.session_state["y_pred"], labels=[0, 1])
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

        # Display important metrics in the sidebar
        important_metrics = [
            ("Sensitivity", sensitivity),
            ("Specificity", specificity),
        ]
        imp_container = st.sidebar.empty()
        with imp_container.container():
            for metric_name, metric_value in important_metrics:
                st.markdown(f"<div class='important-metrics'>{metric_name}: {metric_value:.2f}</div>", unsafe_allow_html=True)

        # Create a DataFrame for the confusion matrix
        data = {
            'Predicted +': [f"{TP} (TP)", f"{FP} (FP)"],
            'Predicted -': [f"{FN} (FN)", f"{TN} (TN)"],
        }
        index_labels = ["Actual +", "Actual -"]
        df = pd.DataFrame(data, index=index_labels)

        # Style the DataFrame using .applymap
        styled_df = df.style.applymap(color_cells)
        st.sidebar.write(styled_df.to_html(), unsafe_allow_html=True)

        # Display normal metrics in the sidebar
        performance_metrics = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("F1 Score", 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0),
        ]
        normal_container = st.sidebar.empty()
        time.sleep(0.5)
        with normal_container.container():
            for metric_name, metric_value in performance_metrics:
                st.markdown(f"<div class='normal-metrics'>{metric_name}: {metric_value:.2f}</div>", unsafe_allow_html=True)

    else:
        # Display important metrics in the sidebar
        important_metrics = [
            ("Sensitivity", "N/A"),
            ("Specificity", "N/A"),
        ]
        imp_container = st.sidebar.empty()
        with imp_container.container():
            for metric_name, metric_value in important_metrics:
                st.markdown(f"<div class='important-metrics'>{metric_name}: {metric_value}</div>", unsafe_allow_html=True)

        # Create a DataFrame for the confusion matrix
        data = {
            'Predicted +': [f"NA (TP)", f"NA (FP)"],
            'Predicted -': [f"NA (FN)", f"NA (TN)"],
        }
        index_labels = ["Actual +", "Actual -"]
        df = pd.DataFrame(data, index=index_labels)

        # Style the DataFrame using .applymap
        styled_df = df.style.applymap(color_cells)
        st.sidebar.write(styled_df.to_html(), unsafe_allow_html=True)

        # Display normal metrics in the sidebar
        performance_metrics = [
            ("Accuracy", "N/A"),
            ("Precision", "N/A"),
            ("F1 Score", "N/A"),
        ]
        normal_container = st.sidebar.empty()
        with normal_container.container():
            for metric_name, metric_value in performance_metrics:
                st.markdown(f"<div class='normal-metrics'>{metric_name}: {metric_value}</div>", unsafe_allow_html=True)

def handle_feedback(index):
    # Check the feedback value stored in session state for thumbs feedback
    feedback_value = st.session_state.get(f"feedback_{index}")
    if feedback_value == 1:
        update_likes(index)
    elif feedback_value == 0:
        update_dislikes(index)
    else:
        update_metrics()
def update_likes(index):
    """Updates metrics when a response is liked."""
    previous_rating = st.session_state['rated_responses'].get(index)
    if previous_rating != 'liked':
        if previous_rating == 'disliked':
            st.session_state['num_incorrect_answers'] -= 1
            st.session_state["y_pred"].remove(0)
        st.session_state['num_correct_answers'] += 1
        st.session_state['user_engagement']['likes'] += 1
        st.session_state['rated_responses'][index] = 'liked'
        st.session_state["y_pred"].append(1)

def update_dislikes(index):
    """Updates metrics when a response is disliked."""
    previous_rating = st.session_state['rated_responses'].get(index)
    if previous_rating != 'disliked':
        if previous_rating == 'liked':
            st.session_state['num_correct_answers'] -= 1
            st.session_state["y_pred"].remove(1)
        st.session_state['num_incorrect_answers'] += 1
        st.session_state['user_engagement']['dislikes'] += 1
        st.session_state['rated_responses'][index] = 'disliked'
        st.session_state["y_pred"].append(0)
        