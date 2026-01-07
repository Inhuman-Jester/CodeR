import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pinecone import Pinecone
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM
import json

load_dotenv(dotenv_path=".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")

index_name = "code-embedding"

os.environ["LANGSMITH_TRACING"] = "true"

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Embedding Model
embedding_model = CodeBERTEmbeddings()

# LLM Model Setup
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

query = "What is the flow of data processing in the codebase?"

# Load Directory Structure & Call Graph
dir_structure = open("database/dir_structure.json", "r").read()
call_graph = open("database/call_graph.json", "r").read()

# prompt for Query Classification
system_prompt = f"""
You are an intelligent query analyzer for a codebase question-answering system.

You are provided with:
1. The directory structure of a codebase
2. The function call graph of the codebase
3. A human query about the codebase

Your tasks are:

1. First, classify the human query into exactly ONE of the following categories:
   - "structural": questions about code structure, files, function calls, execution flow, dependencies, or class relationships
   - "semantic": questions about behavior, purpose, logic, or explanation of code
   - "invalid": queries that do not relate to the codebase

2. After the query is classifie:
   - Determine which file the query is referring to using the directory structure and call graph .
   - Return ONLY valid JSON in the following format:
     {{
       "query_type": "semantic" | "invalid" | "structural",
       "file_path": from directory structure (None if invalid or structural)}}

Rules:
- Do NOT add explanations.
- Do NOT include Markdown, code blocks, or extra text.
- Output must be raw JSON exactly as specified 

Directory Structure:
{dir_structure}

Call Graph:
{call_graph}
"""

messages = [
    ("system", system_prompt),
    ("human", query),
]

print("Classifying Query...")
response = model.invoke(messages)

refined_response = json.loads(response.content)

query_type = refined_response["query_type"]
file_path = refined_response.get("file_path", None)

print(f"Query Type: {query_type}, File Path: {file_path}")

# Syntactical Query Handler
def syntactical_query(query, file_path=file_path):
    embedded_query = embedding_model.embed_documents([query])[0].tolist()
    extracted_code = index.query(
        namespace=file_path,
        vector=embedded_query,
        top_k=3,
        include_metadata=True
    )

    snippets = []

    matches = extracted_code["matches"]
    for i, match in enumerate(matches):
        snippets.append(match['metadata'])


    context = """
    You are a code analysis assistant.

    The incoming query is about the semantics of a specific code file.

    You are provided with :
    1. The directory structure of the codebase
    2. The function call graph of the codebase
    3. A human query
    4. Relevant code snippets from the specified file

    Your task:
    - Use the code snippets to explain the behavior and purpose of the relevant parts of the codebase.
    - Describe how the code works to fulfill the query.
    - Focus only on the components directly involved.

    Rules:
    - Be concise and precise.
    - Do NOT explain unrelated files or functions.
    - Do NOT speculate beyond the given information.
    - Do NOT repeat the code snippets verbatim.

    Directory Structure:
    {dir_structure}

    Call Graph:
    {call_graph}

    Code Snippets from {file_path}:
    {snippets}
    """
    messages = [
        ("system", context),
        ("human", query),
    ]
    response = model.invoke(messages)
    return response.content

    


# Structural Query Handler
def structural_query(query):
    context = f"""
    You are a structural codebase analysis assistant.

    The incoming query is about the structure of the codebase.

    You are provided with:
    1. The directory structure of the codebase
    2. The function call graph of the codebase
    3. A human query

    Your task:
    - Use the directory structure and call graph to explain how the relevant parts of the codebase are connected.
    - Describe the execution or dependency flow related to the query.
    - Focus only on the components directly involved.

    Rules:
    - Be concise and precise.
    - Do NOT explain unrelated files or functions.
    - Do NOT speculate beyond the given information.
    - Do NOT repeat the directory structure or call graph verbatim.

    Directory Structure:
    {dir_structure}

    Call Graph:
    {call_graph}
    """
    messages = [
        ("system", context),
        ("human", query),
    ]

    response = model.invoke(messages)
    return response.content

def route_query_and_generate(query, query_type, file_path):
    print("Generating Response...")
    if query_type == "structural":
        return structural_query(query)
    elif query_type == "semantic":
        return syntactical_query(query, file_path)
    else:
        return None  
    
print(route_query_and_generate(query, query_type, file_path))