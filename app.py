import streamlit as st
import os

# ---- IMPORT YOUR PIPELINES ----
# Replace these with your actual implementations
from source.embedding import CodeIngestionPipeline        # parses + builds call graph + vectors
from source.retrieval_and_generation import rag_chain      # answers queries


code_ingestion_pipeline = CodeIngestionPipeline(
    index_name="code-embedding",
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# --------------------------------
st.set_page_config(
    page_title="Codebase Q&A",
    layout="wide"
)

st.title("CodeR : Structure-Aware Code Retrieval System")

# ---- SESSION STATE INIT ----
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "repo_url" not in st.session_state:
    st.session_state.repo_url = None

# ---- SIDEBAR: REPO INPUT ----
with st.sidebar:
    st.header("📦 Repository")
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repo"
    )

    if st.button("Parse & Index Repository"):
        if not repo_url:
            st.warning("Please enter a repository URL.")
        else:
            with st.spinner("Cloning and parsing repository..."):
                try:
                    code_ingestion_pipeline.run(language="python", repo_url=repo_url)
                    st.session_state.repo_indexed = True
                    st.session_state.repo_url = repo_url
                    st.success("Repository indexed successfully!")
                except Exception as e:
                    st.error(f"Failed to parse repo: {e}")

    st.divider()

    if st.button("🛑 End Session", type="primary"):
        with st.spinner("Cleaning up session..."):
            try:
                code_ingestion_pipeline.delete_index()
            except Exception as e:
                st.error(f"Failed to delete Pinecone index: {e}")

            # Clear Streamlit session
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.success("Session ended. Index deleted.")
            st.rerun()


# ---- MAIN CHAT UI ----
if not st.session_state.repo_indexed:
    st.info("👈 Enter a GitHub repo and click **Parse & Index Repository** to begin.")
    st.stop()

st.subheader("💬 Chat with the Codebase")

# ---- DISPLAY CHAT HISTORY ----
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ---- USER INPUT ----
user_query = st.chat_input("Ask a question about the codebase...")

if user_query:
    # Show user message
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag_chain.invoke({
                    "query": user_query
                })
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer["response"])

    # Save assistant message
    st.session_state.chat_history.append(("assistant", answer["response"]))