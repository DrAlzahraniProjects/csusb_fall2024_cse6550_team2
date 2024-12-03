from constants import *
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import json
import nltk
from web_crawler import merged_data

nltk.download('punkt')

def initialize_milvus(data, milvus_uri=MILVUS_URI):
    """Initialize Milvus, create collection, and insert data from content and internal links."""
    print("Initializing Milvus and creating a collection...")

    # Connect to Milvus
    connections.connect(alias="default", uri=milvus_uri)

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=50000),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000)
    ]
    schema = CollectionSchema(fields, "CSUSB_CSE_Collection")

    # Drop existing collection if present
    collection_name = "CSUSB_CSE_Data"
    if utility.has_collection(collection_name):
        Collection(name=collection_name).drop()

    # Create and load collection
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "FLAT", "metric_type": "L2"})
    collection.load()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def process_entry(entry, results):
        """Process 'content' and 'internal_links' recursively."""
        current_url = entry.get("url", "")

        # Process 'content' fields
        for content in entry.get("content", []):
            if content.get("type") in ["h1", "h2", "h3", "p"]:
                text = content.get("text")
                if text:
                    embedding = model.encode(text).tolist()
                    results.append((text, current_url, embedding))

        # Process 'data' and 'url' fields in `internal_links`
        for link in entry.get("internal_links", []):
            link_url = link.get("url")
            for data_entry in link.get("data", []):
                text = json.dumps(data_entry)  # Convert structured data to string
                embedding = model.encode(text).tolist()
                results.append((text, link_url, embedding))

            # Recurse into nested `internal_links`
            process_entry(link, results)

    # Process all data
    results = []
    for item in data:
        process_entry(item, results)

    # Debugging: Check results
    print(f"Results Count: {len(results)}")
    if not results:
        print("No data to insert. Please check the input data.")
        return

    # Prepare data for insertion
    embeddings, text_contents, urls = [], [], []
    for idx, (text, url, embedding) in enumerate(results):
        # Validate embedding dimensions
        if len(embedding) != 384:
            print(f"Skipping entry at index {idx} with invalid embedding dimension: {len(embedding)}")
            continue
        # Validate text and URL
        if not isinstance(text, str) or not text.strip():
            print(f"Skipping entry at index {idx} with invalid text content: {text}")
            continue
        if not isinstance(url, str) or not url.strip():
            print(f"Skipping entry at index {idx} with invalid URL: {url}")
            continue

        embeddings.append(embedding)
        text_contents.append(text)
        urls.append(url)

    # Debugging: Check final lengths
    print(f"Embeddings: {len(embeddings)}, Texts: {len(text_contents)}, URLs: {len(urls)}")
    if len(embeddings) != len(text_contents) or len(embeddings) != len(urls):
        print("Error: Mismatch in data lengths.")
        return

    # Insert into Milvus
    try:
        collection.insert([embeddings, text_contents, urls])
        print(f"Number of Documents in Collection: {collection.num_entities}")
        print("Data insertion completed.")
    except Exception as e:
        print(f"Error inserting data into Milvus: {e}")


def initialize_milvus_insert_data():
    KB = merged_data()
    print("Input Data Sample:", KB[:3])
    return initialize_milvus(KB)