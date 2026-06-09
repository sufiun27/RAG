import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "./data"
DB_DIR = "./chroma_db"

def embed_and_store_documents():
    print("Step 1: Loading documents...")
    loaders = {
        ".pdf": DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ".txt": DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader),
        ".docx": DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader),
    }
    
    documents = []
    for ext, loader in loaders.items():
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} files with extension {ext}")
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    if not documents:
        print("No documents found.")
        return

    print(f"\nStep 2: Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("\nStep 3: Generating local embeddings via Sentence-Transformers...")
    # This will download the model locally on your first run
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Save to your local Chroma DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Success! Local Vector database saved at: {DB_DIR}")
    return vector_db

if __name__ == "__main__":
    db = embed_and_store_documents()