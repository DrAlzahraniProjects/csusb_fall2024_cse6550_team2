from sentence_transformers import SentenceTransformer

# A cache to store computed embeddings for faster lookups
embedding_cache = {}

# Initialize the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any other model as needed

def emb_text(text: str):
    """
    Generate or retrieve cached embeddings for the provided text using SentenceTransformer.

    Args:
        text (str): The input text to generate embeddings for.

    Returns:
        List[float]: The embedding vector for the input text.
    """
    # Check if the text is already cached
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        # Generate embeddings using the Hugging Face SentenceTransformer
        embedding = embedding_model.encode(text).tolist()
        # Cache the generated embedding for future use
        embedding_cache[text] = embedding
        return embedding
