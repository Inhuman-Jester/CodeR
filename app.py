import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
# from source.retrieval_and_generation import rag_chain 
from dotenv import load_dotenv
from source.embedding import CodeIngestionPipeline

# Load API keys from environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "code-embedding"
index = pc.Index(index_name)

code_ingestion_pipeline = CodeIngestionPipeline(
    index_name="code-embedding",
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Streamlit UI
st.title("CodeR : Structure-Aware Code Retrieval System")# Title of the Application
st.write("Give the link of your git repository.")

# User Input
repo_url = st.text_input("Enter the url:")

if st.button("Get Answer"):# If this button is clicked, then the below code is run
    if repo_url.strip():# strip is method used to remove leading, trailing spaces
        # Run the ingestion pipeline
        code_ingestion_pipeline.run(language="python", repo_url=repo_url)
        
        # Display the result
        st.subheader("Ingestion Status:")
        st.write("Ingestion completed.")
    else:# If the user clicked on the button, without any question, then the below code is run
        st.warning("Please enter a valid repo.")