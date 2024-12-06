import os
import httpx
import numpy as np
import re
from pymilvus import Collection
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from urllib.parse import urlparse

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_api_key():
    """Retrieve the API key from the environment."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Ensure the API key is set in main.py before proceeding.")
    return api_key

def search_milvus(query):
    """Search Milvus collection for a query and return the top results."""
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Load the collection
    collection_name="CSUSB_CSE_Data"
    collection = Collection(name=collection_name)
    collection.load()

    # Convert the query into an embedding
    query_embedding = np.array(model.encode(query), dtype=np.float32).tolist()

    # Define search parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    # Perform the search
    # print("Searching for:", query)
    # Perform the search
    # print("Searching for:", query)

    results = collection.search(
        data=[query_embedding],          # Query embedding
        anns_field="embedding",          # Field to search
        param=search_params,
        limit=50,                         # Number of results
        expr=None,                       # Optional filter
        output_fields=["text_content", "url"]  # Specify fields to retrieve
    )
    # Collect the context
    context_chunks = []

    # Display results
    for i, result in enumerate(results[0]):
        # print(f"Result {i+1}:")
        text_content = result.entity.get("text_content")
        url = result.entity.get("url")
        if text_content:  # Ensure the content is valid
            context_chunks.append(f"{text_content.strip()}\n(Source: {url})")
        # print(f"Text: {text_content}")
        # print(f"URL: {url}")
        # print(f"Score: {result.distance}")
        # print("-" * 40)

    # Create the context by concatenating the top results
    context = " ".join(context_chunks[:30])
    print(context,"Context")
    return context

def extract_keywords_from_query(query, max_keywords=5):
    """Extract keywords dynamically from the query."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_keywords)
    vectorizer.fit([query])  # Fit the vectorizer only on the query
    keywords = vectorizer.get_feature_names_out()
    return list(keywords)  # Ensure keywords are returned as a Python list

def compare_keywords_with_context(query, context, max_keywords=5):
    """Extract keywords from query and compare them with the context."""
    # Extract keywords from the query
    keywords = extract_keywords_from_query(query, max_keywords=max_keywords)

    # Compare keywords with the context
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in context.lower()]

    # Calculate the relevance score
    relevance_score = len(matched_keywords) / len(keywords) if keywords else 0
    return keywords, matched_keywords, relevance_score

def handle_stopword_prompts(query):
    """
    Handle conversational prompts or queries with stop words
    and return a predefined guidance response.
    """
    conversational_prompts = [
        "hi", "hello", "what is your name", "who are you", 
        "how are you", "what do you do", "what's your name"
    ]
    # Normalize the query for comparison
    normalized_query = query.strip().lower()

    if any(prompt in normalized_query for prompt in conversational_prompts):
        return (
            "I am an academic advisor chatbot, designed to assist with CSE-related questions. "
            "I am equipped with data from:\n"
            "- CSE Website: https://www.csusb.edu/cse\n"
            "- CSE Catalog: https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/"
        )
    return None

def get_relevant_context(query):
    """
    Retrieve relevant context and handle low relevance scores or unexpected errors gracefully.
    """
    try:
        # Handle conversational prompts
        guidance_response = handle_stopword_prompts(query)
        if guidance_response:
            return guidance_response, None  # Return guidance directly for stopword prompts

        # Proceed with Milvus search if not a conversational prompt
        context = search_milvus(query)

        # Extract and compare keywords
        keywords, matched_keywords, relevance_score = compare_keywords_with_context(query, context)

        print(f"Keywords: {keywords}")
        print(f"Matched Keywords: {matched_keywords}")
        print(f"Relevance Score: {relevance_score:.2f}")

        # Handle relevance score
        if relevance_score <= 0.33:
            context = (
                "Sorry, I can’t help with that. I’m here to assist with CSE academic advising—"
                "try asking about courses, schedules, or resources!"
            )
            sources = None  # No sources for low relevance
        else:
            # Extract all sources from the context
            sources = []
            for line in context.split("\n"):
                if "(Source:" in line:
                    source = line.split("(Source:")[1].strip().rstrip(")")
                    sources.append(source)

            # Join sources into a single string for further processing
            sources = "\n".join(sources) if sources else None

        return context, sources

    except ValueError as e:
        # Handle the empty vocabulary error gracefully
        if "empty vocabulary" in str(e):
            print(f"Encountered ValueError: {e}")
            context = (
                "I am an academic advisor chatbot, designed to assist with CSE-related questions. "
                "I am equipped with data from:\n"
                "- CSE Website: https://www.csusb.edu/cse\n"
                "- CSE Catalog: https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/"
            )
            return context, None

        # Reraise other unexpected errors
        raise e

def generate_response_with_source(rag_chain, context_chunks, sources, query):
    """
    Generate the final response with the very first source or fallback response,
    ensuring the response text does not include URLs and the source is shown separately.
    """
    # Handle guidance response directly
    guidance_response = handle_stopword_prompts(query)
    if guidance_response:
        return guidance_response  # Return guidance for stopword prompts

    # Initialize variables
    normalized_sources = []

    # Parse sources and extract URLs
    if sources:
        for line in sources.split("\n"):
            if "http" in line:
                # Extract URL and clean it
                url = line.split()[0].rstrip(")")
                parsed_url = urlparse(url)
                normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                normalized_sources.append(normalized_url)

        # Debugging: Check normalized sources
        print("Normalized Sources:", normalized_sources)

        # Get the very first source
        first_source = normalized_sources[0] if normalized_sources else None
    else:
        first_source = None

    # Handle the response based on the source and RAG chain output
    if first_source is None:
        # Low relevance or no sources available
        response = context_chunks  # Fallback response
    else:
        # Generate the response using the RAG chain
        response = rag_chain.invoke({"context": context_chunks, "question": query})

        # Check for URLs in the response text
        urls_in_response = re.findall(r"http[s]?://\S+", response)

        # If the response mentions URLs, remove them
        if urls_in_response:
            for url in urls_in_response:
                response = response.replace(url, "").strip()

        # If the response indicates insufficient information, remove the source
        if response.strip().lower().startswith("i don't"):
            response = f"{response.strip()}"
            first_source = None  # Set source to None
        else:
            # Append the first source to the response
            response = (
                f"{response.strip()}\n\nSource:\n{first_source.strip()}"
            )

    return response

def invoke_llm_for_response(query):
    try:
        """Generate a response with highlighted keywords and exclude sources if no information is provided."""
        llm = ChatMistralAI(model='open-mistral-7b', api_key=get_api_key())
        # Define the prompt template
        PROMPT_TEMPLATE = """
        You are a helpful assistant tasked with answering questions based strictly on the provided context. Use only the information, facts, and statistics explicitly given in the context to formulate your response. Do not include any additional information or assumptions outside the context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Instructions:
        - Provide a concise and accurate answer based solely on the context above.
        - If the context does not contain enough information to answer the question, respond with:
        "I don’t have enough information to answer this question."
        - Do not generate, assume, or make up any details beyond the given context.
        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE
        )
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        context_chunks, source = get_relevant_context(query)

         # Generate the response with the most repetitive source
        response = generate_response_with_source(rag_chain, context_chunks, source, query)

        print(response, "Response")
        return response

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment before trying again."
        else:
            raise e
