import os
import httpx
import numpy as np
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
from nemo_guardrails import guard_rails

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
        limit=30,                         # Number of results
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

def get_relevant_context(query):
    """Retrieve relevant context and handle low relevance scores."""
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

def generate_response_with_source(rag_chain, context_chunks, sources, query):
    """
    Generate the final response with the most repetitive source or fallback response.
    """
    # Normalize and count the occurrences of each source
    if sources:
        # Extract URLs from sources and normalize them
        normalized_sources = []
        for line in sources.split("\n"):
            if "http" in line:
                # Extract and normalize the URL
                url = line.split()[0].rstrip(")")
                parsed_url = urlparse(url)
                normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                normalized_sources.append(normalized_url)

        # Debugging: Check normalized sources
        print("Normalized Sources:", normalized_sources)

        # Count the occurrences of normalized sources
        source_counts = Counter(normalized_sources)

        # Debugging: Check source counts
        print("Source Counts:", source_counts)

        # Get the most repetitive source
        most_repetitive_source = max(source_counts, key=source_counts.get)
    else:
        most_repetitive_source = None

    # Handle the response based on the source and RAG chain output
    if most_repetitive_source is None:
        # Low relevance or no sources available
        response = context_chunks  # Fallback response
    else:
        # Generate the response using the RAG chain
        response = rag_chain.invoke({"context": context_chunks, "question": query})

        # If the response indicates insufficient information, remove the source

        # if "I don't have enough information to answer this question." in response:
        #     response = f"{response.strip()}"

        if response.strip().lower().startswith("i don't"):
            response = f"{response.strip()}"
            # most_repetitive_source = None  # Set source to None
        else:
            # Append the most repetitive source to the response
            response = f"{response.strip()}\n\nSource:\n{most_repetitive_source.strip()}"

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
        guard_rail_chain = guard_rails | rag_chain
        
        context_chunks, source = get_relevant_context(query)

        # Generate the response with the most repetitive source
        response = generate_response_with_source(guard_rail_chain, context_chunks, source, query)

        print(response, "Response")
        return response

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment before trying again."
        else:
            raise e