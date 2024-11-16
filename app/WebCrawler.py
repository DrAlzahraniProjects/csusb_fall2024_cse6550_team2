import json
import time
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests

# Base configuration
base_url = "https://www.csusb.edu"
start_url = f"{base_url}/cse"
MILVUS_URI = "milvus_vector.db"
data = []

def scrape_page(url, section_name):
    """Scrape individual page and add to data list."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        page_data = {
            "url": url,
            "section": section_name,
            "title": soup.title.string if soup.title else "No title",
            "content": []
        }

        # Extract various content elements
        for tag in ["h1", "h2", "h3", "p", "li", "div"]:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:
                    page_data["content"].append({"type": tag, "text": text})

        # Extract tables
        for table in soup.find_all("table"):
            table_data = []
            for row in table.find_all("tr"):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                page_data["content"].append({"type": "table", "data": table_data})

        # Extract images and SVGs
        for img in soup.find_all("img"):
            src = img.get("src")
            alt = img.get("alt", "No description")
            if src:
                full_url = src if src.startswith("http") else base_url + src
                page_data["content"].append({"type": "image", "alt": alt, "url": full_url})

        # Extract links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if text and (href.startswith("http") or href.startswith("/")):
                full_url = href if href.startswith("http") else base_url + href
                page_data["content"].append({"type": "link", "text": text, "url": full_url})

        data.append(page_data)
        time.sleep(1)

    except Exception as e:
        print(f"Error scraping {url}: {e}")

def scrape_main_page(start_url):
    """Scrape main page and all linked pages in the navigation."""
    visited_links = set()
    response = requests.get(start_url)
    soup = BeautifulSoup(response.text, "html.parser")
    nav_links = soup.select("a[href]")

    for link in nav_links:
        href = link.get("href")
        section_name = link.get_text(strip=True)
        if href and (href.startswith("/cse") or (base_url in href and "cse" in href)):
            full_url = href if href.startswith("http") else base_url + href
            if full_url not in visited_links:
                visited_links.add(full_url)
                print(f"Scraping section '{section_name}' at URL: {full_url}")
                scrape_page(full_url, section_name)

def initialize_milvus(data):
    """Initialize Milvus, create collection, and insert data."""
    print("Initializing Milvus and creating a collection...")
    connections.connect(alias="default", uri=MILVUS_URI)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=50000)
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
        collection.insert([[idx], [embedding], [text_content]])

    print("Data insertion completed.")

def initialize_and_scrape():
    """Wrapper function to perform the entire workflow."""
    # Scrape data
    scrape_main_page(start_url)

    # Save to JSON
    # with open(os.path.join(data_dir, "csusb_cse_data.json"), "w") as json_file:
    #     json.dump(data, json_file, indent=4)
        # print("Data saved to JSON.")

    # Initialize Milvus and insert data
    initialize_milvus(data)

    