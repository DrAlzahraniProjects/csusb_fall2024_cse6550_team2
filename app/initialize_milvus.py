
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from constants import MILVUS_URI
from web_crawler import data_source_1, data_source_2, merge_data_sources

def initialize_milvus(data):
    """Initialize Milvus, create collection, and insert data."""
    print("Initializing Milvus and creating a collection...")
    connections.connect(alias="default", uri=MILVUS_URI)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=50000),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=200)
    ]
    schema = CollectionSchema(fields, "CSUSB_CSE_Collection")

    if utility.has_collection("CSUSB_CSE_Data"):
        Collection(name="CSUSB_CSE_Data").drop()

    collection = Collection(name="CSUSB_CSE_Data", schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "FLAT", "metric_type": "L2"})
    collection.load()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    MAX_TEXT_LENGTH = 50000

    for idx, item in enumerate(data):
        # text_content = " ".join([content.get("text", "") for content in item.get("content", [])])
        text_content = " ".join([
            content.get("text", "").strip() + " " + content.get("data", "").strip()
            for content in item.get("content", [])
        ]).strip()

        text_content = text_content[:MAX_TEXT_LENGTH]
        embedding = model.encode(text_content).tolist()
        url = item["url"]
        collection.insert([[idx], [embedding], [text_content],[url]])

    print("Data insertion completed.")

def initialize_and_scrape():
    """Wrapper function to perform the entire workflow."""
    # Scrape data
    knowledge_base = merge_data_sources(data_source_1, data_source_2)
    if not knowledge_base:
        print("No data was scraped. Please check the scraper.")
        return
   
    # Initialize Milvus and insert data
    initialize_milvus(knowledge_base)