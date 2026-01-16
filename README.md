# CodeR : Structure-Aware Code Retrieval System

## Problem Statement

While learning Retrieval-Augmented Generation (RAG), I noticed how tools like GitHub Copilot can autocomplete code by understanding not just syntax, but the structure and context of an entire codebase. This inspired me to explore a related idea: building an assistant that can analyze an existing codebase and answer questions about it, combining semantic understanding with structural awareness.

CodeR aims to bridge this gap by treating a codebase not as plain text, but as a graph of functions, files, and dependencies, enabling more accurate and context-aware code retrieval and explanations

## Flow

1. **_Codebase Cloning_**

- The user provides a GitHub repository URL.
- The repository is cloned into a public workspace directory for analysis.

2.  **_Codebase Parsing_** :
    The codebase is parsed using language-specific AST tools to extract:

    - Function & class definitions (as semantic chunks)
    - Codebase Structure
    - Function call graph (caller–callee relationships)

    This step converts raw code into a structured representation.

3.  **_Indexing_** :
    - Each file acts as a namespace in Pinecone.
    - Functions and classes are embedded using CodeBERT.
    - Metadata =
    ```json
        {
        "chunk_id": "",
        "type": "",
        "language": "",
        "file": "",
        "start_line": ,
        "end_line":
        }
    ```
4.  **_Query Classification & Routing_** : An LLM (Gemini) is fed the query and the codebase struct, and it classifies incoming query into:

    - Semantic queries
    - Structural queries
    - Code Search queries
    - None / irrelevant

    Also, it returns file\_\_path to which the query is relevant ( if it is semantic or code search)

    The query is routed to the appropriate pipeline.

5.  **_Semantic Query Handling_** :

    - The query is embedded.
    - A vector search is performed over the relevant namespace i.e. the file path.
    - Retrieved code chunks are passed to the LLM for explanation, summarization, or comparison along with Codebase Structure and Function call graph.

6.  **_Structural Query Handling_** :

    - No embeddings involved.
    - Answers are derived from: AST, function call graph & file structure maps

7.  **_Code Search_** :
    - From the file path, get the namespace
    - Vector search using meta data filtering to get the specific function/class asked for
    - Response contains path to the file, the name of the node (function/class/etc), the starting and ending line.

## What more? (yet to implement)

1. An AI based debugger integrated
2. ~~Code search & Discovery (help developers find relevant code quickly)~~
3. ~~A beautiful frontend.~~
4. Expand to other languages. (only python codebase right now)
5. ~~Need to add context window.~~
6. Improve Latency

## On your device

1. Clone the file to your system
2. Set up the following env variables in your .env file:
   - PINECONE_API_KEY = "key from pinecone "
   - LANGSMITH_API_KEY = "from Langchain"
   - LANGSMITH_PROJECT = "CodeR"
   - LANGSMITH_TRACING_V2 = "true"
   - GEMINI_API_KEY = "From aistudio.google.com"
3. Setup a virtualenvironment and install all requirements
4. in terminal, run this command : `streamlit run app.py`
