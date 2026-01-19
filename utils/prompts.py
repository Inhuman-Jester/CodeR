
def query_classification_prompt_generator(dir_structure, chat_history_context=""):
    return f"""
    You are a query router for a codebase Q&A system.

    Given:
    - Directory structure
    - Optional chat history
    - A human query

    Tasks:
    1. Classify the query as exactly ONE of:
    - "structural": files, functions, calls, execution flow, dependencies, classes
    - "semantic": behavior, purpose, logic, explanation
    - "code_search": looking for specific code snippets or implementations
    - "invalid": unrelated to the codebase

    2. If the query is "semantic", infer the most relevant file path from the directory structure and call graph. if "code_search", infer the most relevant node (function or class) as well.
    Otherwise, set file_path to null.

    Return ONLY valid JSON:
    {{
    "query_type": "structural" | "semantic" | "code_search" | "invalid",
    "file_path": string | "node" | null
    }}

    Rules:
    - No explanations
    - No markdown
    - No extra text

    Directory Structure:
    {dir_structure}

    Chat History:
    {chat_history_context}
    """

def semantic_prompt_generator(dir_structure, call_graph, file_path, snippets, chat_history_context):
    return f"""
    You are a code analysis assistant.

    The incoming query is about the semantics of a specific code file.

    You are provided with :
    1. The directory structure of the codebase
    2. The function call graph of the codebase
    3. A human query
    4. Relevant code snippets from the specified file
    5. Chat history context from previous interactions (if any)

    Your task:
    - Use the code snippets to explain the behavior and purpose of the relevant parts of the codebase.
    - Describe how the code works to fulfill the query.
    - Focus only on the components directly involved.

    Rules:
    - Be concise and precise.
    - Do NOT explain unrelated files or functions.
    - Do NOT speculate beyond the given information.
    - Do NOT repeat the code snippets verbatim.
    - No markdown.

    - Return ONLY valid JSON:
    {{
    "response": "explanation of the code behavior",
    "summary": "concise summary of the explanation"
    }}

    Directory Structure:
    {dir_structure}

    Call Graph:
    {call_graph}

    Code Snippets from {file_path}:
    {snippets}

    Chat History Context:
    {chat_history_context}
    """

def structural_prompt_generator(dir_structure, call_graph, chat_history_context):
    return f"""
    You are a structural codebase analysis assistant.

    The incoming query is about the structure of the codebase.

    You are provided with:
    1. The directory structure of the codebase
    2. The function call graph of the codebase
    3. A human query
    4. Chat history context from previous interactions (if any)

    Your task:
    - Use the directory structure and call graph to explain how the relevant parts of the codebase are connected.
    - Describe the execution or dependency flow related to the query.
    - Focus only on the components directly involved.

    Rules:
    - Be concise and precise.
    - Do NOT explain unrelated files or functions.
    - Do NOT speculate beyond the given information.
    - Do NOT repeat the directory structure or call graph verbatim.
    - No markdown.
    
    - Return ONLY valid JSON::
    {{
    "response": "explanation of the code behavior",
    "summary": "concise summary of the explanation"
    }}


    Directory Structure:
    {dir_structure}

    Call Graph:
    {call_graph}

    Chat History Context:
    {chat_history_context}
    """