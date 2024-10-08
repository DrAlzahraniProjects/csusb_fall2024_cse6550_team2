import streamlit as st
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Connect
try:
    connections.connect("default", host="localhost", port="19530")
except Exception  as e:
    print(f"Error connecing to Milvus: {e}")
# Load your data files from the extracted directory
data_directory = 'csusb'  # Change this to your data directory
documents = []
for filename in os.listdir(data_directory):
    if filename.endswith('.html'):  # Assuming the pages are saved as HTML
        with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())

# Initialize the SentenceTransformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a Milvus collection
fields = [
    FieldSchema(name="text", dtype=DataType.STRING, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # Adjust dim according to your model
]
schema = CollectionSchema(fields, description="QA collection")
collection = Collection("qa_collection", schema)

# Insert documents into Milvus
embeddings = model.encode(documents).tolist()
collection.insert([documents, embeddings])

# Changes tab title (Warning: Leave at top)
st.set_page_config(page_title = "Academic Chatbot - Team2")

# CSS styling
with open("assets/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# User input for the question
question = st.text_input("Enter your question:")

# Streamlit app title
st.title("Question Answering System")

# User input for the question
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if question:
        # Embed the question
        question_embedding = model.encode([question]).tolist()

        # Perform a search in Milvus
        search_result = collection.search(
            data=question_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE", "top_k": 5},
            limit=5
        )

        # Display the results
        st.subheader("Results:")
        for hit in search_result[0]:
            st.write(f"Document: {hit.entity.get('text')}")
            st.write(f"Score: {hit.score}")
    else:
        st.warning("Please enter a question.")
