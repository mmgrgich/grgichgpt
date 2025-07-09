# Winery Assistant Starter Template (LangChain + Dropbox)

# 1. Install required packages
# pip install langchain openai chromadb dropbox python-dotenv tiktoken

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")

# --- 1. Dropbox sync (optional placeholder, real API integration recommended)
LOCAL_DOCS_DIR = "./dropbox_sync"
os.makedirs(LOCAL_DOCS_DIR, exist_ok=True)

# --- 2. Load and split documents
def load_documents():
    loader = DirectoryLoader(LOCAL_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- 3. Embed and store in vector DB
def get_vectorstore():
    docs = load_documents()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_store")
    return vectordb

# --- 4. Initialize GPT with Retrieval
def get_chain():
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# --- 5. Simple CLI prompt interface
def main():
    print("\nWinery GPT Assistant (CLI Mode)")
    print("Ask questions based on winery SOPs and data.")
    print("Type 'exit' to quit.\n")

    chain = get_chain()
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = chain.run(query)
        print(f"\nAnswer:\n{response}\n")

if __name__ == "__main__":
    main()
