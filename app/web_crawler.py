corpus_source = "https://www.csusb.edu"

import time
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Base configuration

start_url = f"{corpus_source}/cse"
MILVUS_URI = "milvus_vector.db"

def clean_text(text):
    """Clean text by removing unnecessary whitespace and non-alphanumeric characters."""
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r"[^\w\s.,!?-]", "", text)  # Remove non-alphanumeric characters
    return text.strip()

def scrape_page(url, section_name):
    """Scrape individual page and add to data list."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        page_data = {
            "url": url,
            "section": section_name,
            "title": clean_text(soup.title.string) if soup.title else "No title",
            "content": []
        }

        # Extract various content elements
        for tag in ["h1", "h2", "h3", "p", "li", "div"]:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:
                    cleaned_text = clean_text(text)
                    page_data["content"].append({"type": tag, "text": cleaned_text})

        # Extract tables
        for table in soup.find_all("table"):
            table_data = []
            for row in table.find_all("tr"):
                row_data = [clean_text(cell.get_text(strip=True)) for cell in row.find_all(["th", "td"])]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                page_data["content"].append({"type": "table", "data": table_data})

        # Extract images and SVGs
        for img in soup.find_all("img"):
            src = img.get("src")
            alt = img.get("alt", "No description")
            if src:
                full_url = src if src.startswith("http") else corpus_source + src
                page_data["content"].append({"type": "image", "alt": alt, "url": full_url})

        # Extract links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if text and (href.startswith("http") or href.startswith("/")):
                full_url = href if href.startswith("http") else corpus_source + href
                cleaned_text = clean_text(text)
                page_data["content"].append({"type": "link", "text": cleaned_text, "url": full_url})
        time.sleep(1)
        return page_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")

def scrape_main_page(start_url):
    """Scrape main page and all linked pages in the navigation."""
    visited_links = set()
    data = []  # Collect all scraped data here

    try:
        response = requests.get(start_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        nav_links = soup.select("a[href]")

        for link in nav_links:
            href = link.get("href")
            section_name = link.get_text(strip=True)
            if href and (href.startswith("/cse") or (corpus_source in href and "cse" in href)):
                full_url = href if href.startswith("http") else corpus_source + href
                if full_url not in visited_links:
                    visited_links.add(full_url)
                    # print(f"Scraping section '{section_name}' at URL: {full_url}")
                    page_data = scrape_page(full_url, section_name)
                    if page_data:
                        data.append(page_data)  # Add the scraped page data to the list
    except Exception as e:
        print(f"Error scraping {start_url}: {e}")

    return data  # Return the collected data


def chunk_data(data):
    """
    Split the text content into manageable chunks using RecursiveCharacterTextSplitter.

    Args:
        data (list): List of scraped data dictionaries.

    Returns:
        list: List of chunked data with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunked_data = []

    for page in data:
        combined_text = ""
        for content in page.get("content", []):
            if content["type"] in ["h1", "h2", "h3", "p", "li", "div","table","image"]:
                combined_text += content["text"] + " "

        text_chunks = splitter.split_text(combined_text)

        for chunk in text_chunks:
            chunked_data.append({
                "url": page["url"],
                "section": page["section"],
                "title": page.get("title", "No title"),
                "text_chunk": chunk
            })

    return chunked_data

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
        text_content = " ".join([content.get("text", "") for content in item.get("content", [])])
        text_content = text_content[:MAX_TEXT_LENGTH]
        embedding = model.encode(text_content).tolist()
        url = item["url"]
        collection.insert([[idx], [embedding], [text_content],[url]])

    print("Data insertion completed.")

def initialize_and_scrape():
    """Wrapper function to perform the entire workflow."""
    # Scrape data
    data = scrape_main_page(start_url)
    if not data:
        print("No data was scraped. Please check the scraper.")
        return
    # for item in data:
    #     print(f"URL: {item['url']}")
    # for content in item['content']:
    #     print(f"Type: {content['type']}, Text: {content.get('text', '')}")
    chunked_data = chunk_data(data)
    print(f"Total pages scraped: {len(data)}")
   
    # Initialize Milvus and insert data
    initialize_milvus(chunked_data)
    