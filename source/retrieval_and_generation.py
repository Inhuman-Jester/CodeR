import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pinecone import Pinecone, ServerlessSpec
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import TypedDict, Optional, List
from utils.prompts import query_classification_prompt_generator, semantic_prompt_generator, structural_prompt_generator
from langgraph.graph import StateGraph, END, START
from utils.helper import formatContext

load_dotenv(dotenv_path=".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")

index_name = "code-embedding"

os.environ["LANGSMITH_TRACING"] = "true"

print("Here")

def ensure_index():
        existing = [i["name"] for i in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)

        return pc.Index(index_name)

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = ensure_index()

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# LLM Model Setup
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None
)

# Load Directory Structure & Call Graph
dir_structure = None
call_graph = None

if(os.path.exists("database/dir_structure.json") and os.path.exists("database/call_graph.json")):
    with open("database/dir_structure.json", "r") as f:
        dir_structure = json.load(f)

    with open("database/call_graph.json", "r") as f:
        call_graph = json.load(f)
        
class CodeAgentState(TypedDict):
    query: str
    query_type: Optional[str]
    file_path: Optional[str]
    snippets: Optional[List[dict]]
    response: Optional[str]
    summary: Optional[str]
    chat_history_context: Optional[str]


def retrieve_chat_history(state: CodeAgentState):
    query = state["query"]
    embedded_query = embedding_model.embed_documents(
        [query]
    )[0]

    chat_history_results = index.query(
        vector=embedded_query,
        top_k=2,
        include_metadata=True,
        namespace="chat_history"
    )

    if len(chat_history_results['matches']):
        selected_chat_history_results = [doc for doc in chat_history_results['matches'] if doc.get('score',0) > 0.4]
    else:
        selected_chat_history_results = []

    formatted_chat_history_results = formatContext({'matches': selected_chat_history_results})
    return {"chat_history_context": formatted_chat_history_results}


def classify_query(state: CodeAgentState):
    query_classification_prompt = query_classification_prompt_generator(dir_structure, state.get("chat_history_context",""))
    
    messages = [
        ("system", query_classification_prompt),
        ("human", state["query"]),
    ]
    response = model.invoke(messages)
    result = json.loads(response.content)

    return {
        "query_type": result["query_type"],
        "file_path": result.get("file_path","")
    }


def semantic_retrieval(state: CodeAgentState):
    embedded_query = embedding_model.embed_documents(
        [state["query"]]
    )[0]

    results = index.query(
        namespace=state["file_path"],
        vector=embedded_query,
        top_k=3,
        include_metadata=True
    )

    snippets = [m["metadata"] for m in results["matches"]]

    return {"snippets": snippets}


def semantic_reasoning(state: CodeAgentState):
    semantic_prompt = semantic_prompt_generator(dir_structure, call_graph, state["file_path"], state["snippets"], state.get("chat_history_context",""))

    messages = [
        ("system", semantic_prompt),
        ("human", state["query"]),
    ]

    response = model.invoke(messages)
    result = json.loads(response.content)
    return {"response": result["response"], "summary": result["summary"]}
    

def structural_reasoning(state: CodeAgentState):
    structural_prompt = structural_prompt_generator(dir_structure, call_graph, state.get("chat_history_context",""))

    messages = [
        ("system", structural_prompt),
        ("human", state["query"]),
    ]
    response = model.invoke(messages)
    result = json.loads(response.content)
    return {"response": result["response"], "summary": result["summary"]}


def code_search_reasoning(state: CodeAgentState):
    node_path = state["file_path"]
    namespace = node_path.rsplit("::", 1)[0]

    vector = [0.0] * 384  # Dummy vector for code search reasoning
    results = index.query(
        namespace=namespace,
        vector=vector,
        filter={
            "chunk_id": {"$eq": node_path}
        },
        top_k=3,
        include_metadata=True
    )

    code_location = [m["metadata"] for m in results["matches"]][0]

    response = f"""The code is located in file: `{code_location["file"]}` in {code_location["type"]} : `{code_location["chunk_id"].rsplit("::", 1)[-1]}`, starting at line `{code_location["start_line"]}` to line `{code_location["end_line"]}`."""
    summary = f"Code found in {code_location['chunk_id']} at lines {code_location['start_line']}-{code_location['end_line']}."
    return {"response": response, "summary": summary}


def invalid_handler(state: CodeAgentState):
    return {
        "response": "This query does not relate to the codebase."
    }

def update_chat_history(state: CodeAgentState):
    if(state["query_type"] == "invalid"): # Do not store invalid queries
        return {}
    
    qa_text = f"Q: {state['query']}\nA: {state['summary']}"

    embedding = embedding_model.embed_documents([qa_text])[0]

    index.upsert(
        vectors=[
            {
                "id": str(os.urandom(16).hex()),
                "values": embedding,
                "metadata": {
                    "text": qa_text
                }
            }
        ],
        namespace="chat_history"
    )

    return {}


graph = StateGraph(CodeAgentState)

graph.add_node("retrieve_chat_history", retrieve_chat_history)
graph.add_node("classify", classify_query)
graph.add_node("semantic_retrieval", semantic_retrieval)
graph.add_node("semantic_reasoning", semantic_reasoning)
graph.add_node("structural_reasoning", structural_reasoning)
graph.add_node("code_search_reasoning", code_search_reasoning)
graph.add_node("invalid", invalid_handler)
graph.add_node("update_chat_history", update_chat_history)


def route(state: CodeAgentState):
    if state["query_type"] == "semantic":
        return "semantic"
    elif state["query_type"] == "structural":
        return "structural"
    elif state["query_type"] == "code_search":
        return "code_search"
    else:
        return "invalid"
    
graph.add_edge(START, "retrieve_chat_history")
graph.add_edge("retrieve_chat_history", "classify")
graph.add_conditional_edges(
    "classify",
    route,
    {
        "semantic": "semantic_retrieval",
        "structural": "structural_reasoning",
        "code_search": "code_search_reasoning",
        "invalid": "invalid"
    }
)
graph.add_edge("semantic_retrieval", "semantic_reasoning")
graph.add_edge("semantic_reasoning", "update_chat_history")
graph.add_edge("structural_reasoning", "update_chat_history")
graph.add_edge("code_search_reasoning", "update_chat_history")
graph.add_edge("invalid", "update_chat_history")
graph.add_edge("update_chat_history", END)


rag_chain = graph.compile()

result = rag_chain.invoke({
    "query": "Can you explain the structure of code?"
})

print(result["response"])