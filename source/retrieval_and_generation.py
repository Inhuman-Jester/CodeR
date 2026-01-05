import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pinecone import Pinecone
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM

load_dotenv(dotenv_path=".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")

index_name = "code-embedding"

os.environ["LANGSMITH_TRACING"] = "true"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

embedding_model = CodeBERTEmbeddings()

query = "Which function is responsible for storing chat history?"

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

prompt = """You are a query analyzer for a codebase question-answering system.
    Your task is to classify a human query into exactly ONE of the following categories:

    - "structural": questions about code structure, function calls, execution flow, dependencies, files, or class relationships
    - "semantic": questions about behavior, purpose, logic, or explanation of code

    Examples:
    - "What are the main classes in file X?" → structural
    - "Which function calls Y?" → structural
    - "Explain function Y in file Z" → semantic
    - "What does this codebase do?" → semantic

    Return ONLY valid JSON in the following format:
    {
    "query_type": "structural" | "semantic" | "invalid"
    }
    Do NOT wrap the output in Markdown, code blocks, or backticks.
    Return raw JSON only.

    Do not add any explanation.
"""

messages = [
    ("system", prompt),
    ("human", query),
]

response = model.invoke(messages)

query_type = json.loads(response.content)

def syntactical_query(query_type, query):
    embedded_query = embedding_model.embed_documents([query])[0].tolist()
    extracted_code = index.query(vector=embedded_query, top_k=5, include_metadata=True)

    matches = extracted_code.get("matches", [])
    texts = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        text = meta.get("text") or meta.get("page_content") or meta.get("file") or str(m.get("id", ""))
        texts.append(text)

    return "; ".join(texts)

# print(syntactical_query(query_type, query))
print("Query Type:")

messages = [
    ("human", query+syntactical_query(query_type, query)),
]
response = model.invoke(messages)
print(response.content)