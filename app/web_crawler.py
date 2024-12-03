from constants import *
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

# Website 1 - "https://www.csusb.edu/cse"
def scrape_source_1(base_url):
    """
    Scrape all internal links and their content from the given base URL.
    Detect and parse tables within div elements. Handle accordion cases.
    Only scrape allowed navigation items and exclude specific content.
    Ensure all links are properly formatted as full URLs and duplicates are removed.
    :param base_url: The base URL of the website to scrape.
    :return: A list of scraped data from all internal links.
    """
    visited_links = set()
    scraped_data = []

    def parse_table(table):
        """Parse a table and return structured data as rows with headers."""
        headers = [header.get_text(strip=True) for header in table.find_all("th")]
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) == len(headers):  # Match cells to headers
                row_data = dict(zip(headers, cells))
                rows.append(row_data)
        return rows

    def format_url(href):
        """Ensure the href is a full URL."""
        if href.startswith("http"):
            return href
        return urljoin("https://www.csusb.edu", href.lstrip("/"))

    def scrape_page(url, visited):
        """Scrape a single page and extract its content, including tables within divs."""
        if url in visited:
            return None
        visited.add(url)

        # print(f"Scraping page: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            page_data = {"url": url, "content": []}
            unique_links = set()  # To track unique links on this page

            # Extract navigation links
            nav_links = soup.find_all("a", href=True)
            for link in nav_links:
                link_text = link.get_text(strip=True)
                href = link["href"]

                # Ensure the href is a full URL
                href = format_url(href)

                # Skip duplicates
                if href in unique_links:
                    continue
                unique_links.add(href)

                # Store links
                # print(f"Storing link: {href}")
                page_data["content"].append({"type": "link", "url": href, "text": link_text})

                if link_text in ALLOWED_CSE_NAVIGATION_SECTIONS:  # Only process allowed navigation
                    # print(f"Allowed navigation link found: {link_text} -> {href}")

                    # Recursively scrape allowed navigation links
                    if href.startswith(base_url) and href not in visited:
                        nested_page_data = scrape_page(href, visited)
                        if nested_page_data:
                            page_data.setdefault("internal_links", []).append(nested_page_data)

                # Include PDF links explicitly
                if href.endswith(".pdf"):
                    # print(f"PDF link found: {href}")
                    page_data["content"].append({"type": "pdf", "url": href, "text": link_text})

            # Extract content from <div> and check for tables
            for div in soup.find_all("div"):
                table = div.find("table")  # Check if there's a table inside the div
                if table:
                    # print(f"Table found inside a div on {url}")
                    structured_table = parse_table(table)
                    if structured_table:
                        page_data["content"].append({"type": "table", "data": structured_table})

                # Check if the div contains an accordion
                if "accordion" in div.get("class", []) or "accordion" in div.get("id", ""):
                    for p in div.find_all("p"):
                        a_tag = p.find("a", href=True)
                        if a_tag:
                            href = a_tag["href"]
                            href = format_url(href)

                            # Skip duplicates
                            if href in unique_links:
                                continue
                            unique_links.add(href)

                            # Skip excluded links
                            if href in EXCLUDED_URLS or p.get_text(strip=True) in EXCLUDED_TEXTS:
                                # print(f"Skipping excluded link: {href}")
                                continue

                            accordion_content = {
                                "type": "link",
                                "url": href,
                                "text": p.get_text(strip=True),
                            }
                            # print(f"Accordion link found: URL = {href}, Text = {p.get_text(strip=True)}")
                            page_data["content"].append(accordion_content)

            # Extract other HTML elements
            for tag in ["h1", "h2", "h3", "p", "li"]:
                for element in soup.find_all(tag):
                    text = element.get_text(strip=True)
                    if text and text not in EXCLUDED_TEXTS:  # Skip excluded texts
                        page_data["content"].append({"type": tag, "text": text})

            return page_data

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    # Start scraping from the base URL
    main_page_data = scrape_page(base_url, visited_links)
    if main_page_data:
        scraped_data.append(main_page_data)

    return scraped_data

# Website 2 - "https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/"
def scrape_navigation_section(url, section_name, visited=set()):
    """
    Scrape a specific navigation section and follow internal links.
    :param url: The base URL of the section.
    :param section_name: The name of the navigation section.
    :param visited: Set to track visited links.
    :return: Scraped data from the section without the `section` field.
    """
    try:
        if url in visited:
            return []
        visited.add(url)

        # print(f"Scraping navigation section: {url}")  # Print the navigation section URL
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Create section data, excluding the `section` field
        section_data = {"url": url, "content": []}

        # Scrape main content of the section
        for tag in ["h1", "h2", "h3", "p", "li", "div"]:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:
                    section_data["content"].append({"type": tag, "text": text})

        # Scrape and process internal links within the section
        internal_data = scrape_internal_links(soup, url, visited)
        if internal_data:
            section_data["internal_links"] = internal_data

        return section_data

    except Exception as e:
        print(f"Error scraping section {section_name} at {url}: {e}")
        return None

def scrape_internal_links(soup, base_url, visited):
    """
    Find and scrape data from all internal links within a section.
    :param soup: Parsed HTML content of the current page.
    :param base_url: Base URL of the current page.
    :param visited: Set to track visited links.
    :return: List of scraped content from internal links.
    """
    internal_content = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(base_url, href)
        # Check if it's an internal link and not visited
        if full_url not in visited and full_url.startswith(base_url):
            # print(f"Scraping internal link: {full_url}")  # Print the link being scraped
            visited.add(full_url)
            try:
                response = requests.get(full_url)
                response.raise_for_status()
                sub_soup = BeautifulSoup(response.text, 'html.parser')

                # Scrape content from the internal page
                page_content = []
                for tag in ["h1", "h2", "h3", "p", "li", "div"]:
                    for element in sub_soup.find_all(tag):
                        text = element.get_text(strip=True)
                        if text:
                            page_content.append({"type": tag, "text": text})

                # Check for further internal links within the page
                deeper_links = scrape_internal_links(sub_soup, base_url, visited)
                if deeper_links:
                    page_content.extend(deeper_links)

                internal_content.append({
                    "url": full_url,
                    "content": page_content
                })

            except Exception as e:
                print(f"Error scraping internal link {full_url}: {e}")

    return internal_content


def merge_data_sources(data_source_1, data_source_2):
    """
    Merge two data sources into one unified knowledge base.
    :param data_source_1: Data scraped from the main page.
    :param data_source_2: Data scraped from navigation sections.
    :return: Merged data source.
    """
    merged_data = []

    # Add data from data_source_1
    if isinstance(data_source_1, list):
        merged_data.extend(data_source_1)
    elif isinstance(data_source_1, dict):
        merged_data.append(data_source_1)

    # Add data from data_source_2
    if isinstance(data_source_2, list):
        merged_data.extend(data_source_2)
    elif isinstance(data_source_2, dict):
        merged_data.append(data_source_2)

    return merged_data


data_source_1 = scrape_source_1(CORPUS_SOURCES[0])
data_source_2 = scrape_navigation_section(CORPUS_SOURCES[1],ALLOWED_CATALOG_NAVIGATION_SECTIONS)

def merged_data():
    return merge_data_sources(data_source_1, data_source_2)
  
# USE THIS FUNCTION TO SAVE THE SCRAPED DATA TO A JSON FILE

# def save_to_json(file_name, data):
#     """
#     Save the scraped data to a JSON file.
#     :param file_name: Name of the JSON file.
#     :param data: Scraped data to save.
#     """
#     try:
#         with open(file_name, 'w', encoding='utf-8') as json_file:
#             json.dump(data, json_file, ensure_ascii=False, indent=4)
#         # print(f"Data successfully saved to {file_name}")
#     except Exception as e:
#         print(f"Error saving data to {file_name}: {e}")

# Use the function below to save the scraped data to a JSON file
# save_to_json("cse_data.json",knowledge_base)
