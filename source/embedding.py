# Pinecone Setup
import os
# Force CPU-only to avoid torch attempting unsupported accelerators in hosted environments.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langsmith import utils
from langchain_core.documents import Document
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM
from source.preprocessing import extract_spaces
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv(".env")

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")


os.environ["LANGCHAIN_TRACING_V2"] = "true"

utils.tracing_is_enabled()

class CodeIngestionPipeline:
    def __init__(
        self,
        index_name = "code-embedding",
        pinecone_api_key = PINECONE_API_KEY,
        embedding_model= None,
    ):
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        def _build_embedding_model():
            common_kwargs = {
                "model_name": "all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True},
            }

            try:
                return HuggingFaceEmbeddings(**common_kwargs)
            except NotImplementedError:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

                try:
                    import torch

                    torch.set_default_device("cpu")
                except Exception:
                    pass

                return HuggingFaceEmbeddings(**common_kwargs)

        self.embedding_model = embedding_model or _build_embedding_model()

        self.index = None

    # Stage 1 
    def extract(self, language: str, repo_url: str):
        logging.info("Stage 1: Extracting code spaces...")
        return extract_spaces(language=language, repo_url=repo_url)

    # Stage 2 
    def ensure_index(self):
        existing = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            logging.info(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)

        self.index = self.pc.Index(self.index_name)

    # Stage 3 
    def build_documents(self, spaces: dict):
        logging.info("Stage 3: Building LangChain Documents...")
        documents = {}

        for path, elements in spaces.items():
            docs = []
            for element in elements:
                docs.append(
                    Document(
                        page_content=element["code"],
                        metadata={
                            "chunk_id": str(element["id"]),
                            "type": element["metadata"]["type"],
                            "language": element["metadata"]["language"],
                            "file": element["metadata"]["file"],
                            "start_line": element["metadata"]["start_line"],
                            "end_line": element["metadata"]["end_line"],
                        }
                    )
                )
            documents[path] = docs

        return documents

    # Stage 4 
    def ingest(self, documents: dict):
        logging.info("Stage 4: Uploading vectors to Pinecone...")

        for namespace, docs in documents.items():
            PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                index_name=self.index_name,
                namespace=namespace
            )

    # Orchestration
    def run(self, language: str, repo_url: str):
        spaces = self.extract(language, repo_url)
        self.ensure_index()
        documents = self.build_documents(spaces)
        self.ingest(documents)

        logging.info("Code ingestion pipeline completed.")

    def delete_index(self):
        existing_indexes = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)