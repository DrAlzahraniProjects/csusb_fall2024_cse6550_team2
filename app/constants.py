CORPUS_SOURCES = ["https://www.csusb.edu/cse","https://catalog.csusb.edu/colleges-schools-departments/natural-sciences/computer-science-engineering/"]

ALLOWED_CSE_NAVIGATION_SECTIONS = [
    "Welcome",
    "Programs",
    "Faculty and Staff",
    "Advising",
    "Resources",
    "Internships & Careers",
    "Computer Labs & Support",
    "Faculty in the News",
    "Contact Us",
]

# Define exclusions
EXCLUDED_TEXTS = ["Give to CNS"]  # Keywords to exclude
EXCLUDED_URLS = ["https://www.csusb.edu/give-cns"]  # Specific URLs to exclude

ALLOWED_CATALOG_NAVIGATION_SECTIONS = ["Overview", "Faculty", "Undergraduate Degrees", "Graduate Degree", "Minor", "Certificates", "Courses"]


# os.makedirs("milvus_lite", exist_ok=True)
MILVUS_URI = "milvus_vector.db"