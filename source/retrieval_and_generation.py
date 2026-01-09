import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pinecone import Pinecone
from model.codeBERT import CodeBERTEmbeddings, EMBEDDING_DIM

from typing import TypedDict, Optional, List
from utils.prompts import query_classification_prompt_generator, semantic_prompt_generator, structural_prompt_generator
from langgraph.graph import StateGraph, END, START

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

# Load Directory Structure & Call Graph
dir_structure = open("database/dir_structure.json", "r").read()
call_graph = open("database/call_graph.json", "r").read()

query = "What is the flow of data processing in the codebase?"

class CodeAgentState(TypedDict):
    query: str
    query_type: Optional[str]
    file_path: Optional[str]
    snippets: Optional[List[dict]]
    response: Optional[str]


def classify_query(state: CodeAgentState):
    query_classification_prompt = query_classification_prompt_generator(dir_structure, call_graph)
    
    messages = [
        ("system", query_classification_prompt),
        ("human", state["query"]),
    ]
    response = model.invoke(messages)
    result = json.loads(response.content)

    return {
        "query_type": result["query_type"],
        "file_path": result.get("file_path")
    }


def semantic_retrieval(state: CodeAgentState):
    embedded_query = embedding_model.embed_documents(
        [state["query"]]
    )[0].tolist()

    results = index.query(
        namespace=state["file_path"],
        vector=embedded_query,
        top_k=3,
        include_metadata=True
    )

    snippets = [m["metadata"] for m in results["matches"]]

    return {"snippets": snippets}


def semantic_reasoning(state: CodeAgentState):
    semantic_prompt = semantic_prompt_generator(dir_structure, call_graph, state["file_path"], state["snippets"])

    messages = [
        ("system", semantic_prompt),
        ("human", state["query"]),
    ]

    response = model.invoke(messages)
    return {"response": response.content}
    

def structural_reasoning(state: CodeAgentState):
    structural_prompt = structural_prompt_generator(dir_structure, call_graph)

    messages = [
        ("system", structural_prompt),
        ("human", state["query"]),
    ]
    response = model.invoke(messages)
    return {"response": response.content}


def invalid_handler(state: CodeAgentState):
    return {
        "response": "This query does not relate to the codebase."
    }

graph = StateGraph(CodeAgentState)

graph.add_node("classify", classify_query)
graph.add_node("semantic_retrieval", semantic_retrieval)
graph.add_node("semantic_reasoning", semantic_reasoning)
graph.add_node("structural_reasoning", structural_reasoning)
graph.add_node("invalid", invalid_handler)


def route(state: CodeAgentState):
    if state["query_type"] == "semantic":
        return "semantic_retrieval"
    elif state["query_type"] == "structural":
        return "structural_reasoning"
    else:
        return "invalid"
    
graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route,
    {
        "semantic_retrieval": "semantic_retrieval",
        "structural_reasoning": "structural_reasoning",
        "invalid": "invalid"
    }
)
graph.add_edge("semantic_retrieval", "semantic_reasoning")
graph.add_edge("semantic_reasoning", END)
graph.add_edge("structural_reasoning", END)
graph.add_edge("invalid", END)


app = graph.compile()

result = app.invoke({
    "query": "What is the flow of data processing in the codebase?"
})

print(result["response"])