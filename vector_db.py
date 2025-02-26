import os
import hashlib
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

def hash_file_content(filepath):
    """Generate a hash for the content of a file."""
    with open(filepath, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()

def create_or_update_vector_db():
    """Create or update the vector database with new PDFs."""
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Check if the vector database already exists
    if os.path.exists('./chroma_db'):
        print("Vector database already exists. Checking for new PDFs...")
        vector_db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
        
        # Load existing metadata
        existing_metadata = {doc.metadata.get("file_hash", "") for doc in vector_db.similarity_search("", k=vector_db._collection.count())}

        # Load new documents
        loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Filter out already processed files
        new_docs = []
        for doc in documents:
            file_hash = hash_file_content(doc.metadata["source"])
            if file_hash not in existing_metadata:
                doc.metadata["file_hash"] = file_hash  # Add hash to metadata
                new_docs.append(doc)
        
        # Add only new documents to the vector DB
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text = text_splitter.split_documents(new_docs)
            vector_db.add_documents(text)
            vector_db.persist()
            print(f"Added {len(new_docs)} new documents to the vector database.")
        else:
            print("No new documents to add.")
    else:
        print("Vector database does not exist. Creating a new one...")
        loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(text, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        print("ChromaDB created and data saved.")
    
    return vector_db

if __name__ == "__main__":
    create_or_update_vector_db()