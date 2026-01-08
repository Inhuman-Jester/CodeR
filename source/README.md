# CodeR : Structure-Aware Code Retrieval System

## Problem Statement

While learning Retrieval-Augmented Generation (RAG), I noticed how tools like GitHub Copilot can autocomplete code by understanding not just syntax, but the structure and context of an entire codebase. This inspired me to explore a related idea: building an assistant that can analyze an existing codebase and answer questions about it, combining semantic understanding with structural awareness.

CodeR aims to bridge this gap by treating a codebase not as plain text, but as a graph of functions, files, and dependencies, enabling more accurate and context-aware code retrieval and explanations

## Flow

1. **_Codebase Cloning_**

- The user provides a GitHub repository URL.
- The repository is cloned into a public workspace directory for analysis.

2.  **_Codebase Parsing_** :
    The codebase is parsed using language-specific AST tools to extract: - Function & class definitions (as semantic chunks) - Codebase Structure - Function call graph (caller–callee relationships)

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

    - Sytactical queries
    - Structural queries
    - None / irrelevant

    Also, it returns file\_\_path to which the query is relevant (only if it is syntactical)

    The query is routed to the appropriate pipeline.

5.  **_Syntactical Query Handling_** :

    - The query is embedded.
    - A vector search is performed over the relevant namespace i.e. the file path.
    - Retrieved code chunks are passed to the LLM for explanation, summarization, or comparison along with Codebase Structure and Function call graph.

6.  **_Structural Query Handling_** :
    - No embeddings involved.
    - Answers are derived from: AST, function call graph & file structure maps

## What more?

1. An AI based debugger integrated
2. Code search & Discovery (help developers find relevant code quickly)
3. A beautiful frontend.
4. Expand to other languages.
