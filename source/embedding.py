import os
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langsmith import utils
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM
from source.preprocessing import extract_spaces

# Get Code Chunks
spaces = extract_spaces(
    language="python"
)

print(f"Extracted {len(spaces)} code chunks.")
# Environment & Logging
load_dotenv(".env")

logging.basicConfig(level=logging.INFO)

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

# LangSmith Tracing Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

utils.tracing_is_enabled()

# Pinecone Setup
INDEX_NAME = "code-embedding"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Index Exists
def ensure_index():
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        logging.info(f"Index '{INDEX_NAME}' already exists.")
        return

    logging.info(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

ensure_index()
index = pc.Index(INDEX_NAME)

# Embedding Model
embedding_model = CodeBERTEmbeddings()

# Vector Ingestion
logging.info("Uploading documents to Pinecone...")

documents = {}

for path, elements in spaces.items():
    document = []
    for element in elements:
            document.append(
            Document(
                page_content=element["code"],
                metadata={
                    "chunk_id": str(element["id"]),
                    "type": str(element["metadata"]["type"]),
                    "language": str(element["metadata"]["language"]),
                    "file": str(element["metadata"]["file"]),
                    "start_line": int(element["metadata"]["start_line"]),
                    "end_line": int(element["metadata"]["end_line"]),
                }
            )
        )
    documents[path] = document
    

# Create Vector Store and Add Documents
for path, docs in documents.items():
    vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=CodeBERTEmbeddings(),
    index_name=INDEX_NAME,
    namespace=path
)


logging.info("Ingestion complete.")