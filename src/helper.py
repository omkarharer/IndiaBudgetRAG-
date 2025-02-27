import os
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
# import pinecone
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

# Initialize LLM
OPENAI_API_KEY = os.environ.get('groq')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings model
def load_embedding():
    """Load the HuggingFace embeddings model."""
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Load vector database
def load_vector_db():
    """Load the Chroma vector database."""
    embeddings = load_embedding()
    vector_db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
    return vector_db

# Load LLM
def load_llm():
    """Load the Groq LLM."""
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=OPENAI_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Load memory for chat history
def load_memory(llm):
    """Load conversation memory."""
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    return memory

# Set up QA chain
def setup_qa_chain(vector_db, llm, memory):
    """Set up the ConversationalRetrievalChain with a custom prompt template."""
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    
    # Custom prompt template for the budget chatbot
    prompt_templates = """You are an expert financial analyst specializing in the Indian Budget. Provide clear, insightful, and data-driven responses to the following question:  
    Context: {context}  
    User: {question}  
    BudgetBot:"""
    
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    
    # Create ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# def pinecone_setup():
#     import pinecone
#     embeddings = load_embedding()
#     pinecone_api_key  = os.environ.get('pinecone')
#     pc = Pinecone(
#         api_key=pinecone_api_key 
#     )
#     index_name = "rag"
#     # Connect to the existing index
#     index = pc.Index(index_name)
#     # # Initialize Pinecone vector store
#     docsearch = Pinecone(index, embeddings.embed_query, "text")
#     return docsearch
    
def pinecone_setup():
    """Initialize Pinecone and connect to the existing index."""
    # Load Pinecone API key from environment variables
    pinecone_api_key = os.environ.get('pinecone')
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found in environment variables.")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Define the index name
    index_name = "rag"  # Replace with your Pinecone index name
    
    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist in your Pinecone project.")
    
    # Connect to the existing index
    index = pc.Index(index_name)
    
    # Load embeddings model
    embedding_model = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize Pinecone vector store
    # docsearch = LangchainPinecone(index, embedding_model.embed_query, "text")
    docsearch = LangchainPinecone.from_existing_index(index_name, embedding_model)
    
    return docsearch


def pinecone_setup_new():
    """Initialize Pinecone and connect to the existing index."""
    # Load Pinecone API key from environment variables
    try:
        # For Streamlit deployment
        import streamlit as st
        pinecone_api_key = st.secrets["pinecone"]
    except:
        # For local development
        from dotenv import load_dotenv
        load_dotenv()
        pinecone_api_key = os.environ.get('pinecone')
    
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found in environment variables.")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Define the index name
    index_name = "rag"  # Replace with your Pinecone index name
    
    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist in your Pinecone project.")
    
    # Load embeddings model
    embedding_model = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize Pinecone vector store
    try:
        # Try the direct initialization approach
        index = pc.Index(index_name)
        docsearch = LangchainPinecone(index, embedding_model.embed_query, "text")
    except Exception as e:
        # Fallback to the from_existing_index approach
        print(f"Direct initialization failed: {e}. Falling back to from_existing_index.")
        docsearch = LangchainPinecone.from_existing_index(index_name, embedding_model)
    
    return docsearch



# Set up QA chain for Pinecone
def setup_qa_chain_pinecone(docsearch, llm, memory):
    """Set up the ConversationalRetrievalChain with a custom prompt template for Pinecone."""
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    
    # Custom prompt template for the budget chatbot
    prompt_templates = """You are an expert financial analyst specializing in the Indian Budget. Provide clear, insightful, and data-driven responses to the following question:  
    Context: {context}  
    User: {question}  
    BudgetBot:"""
    
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    
    # Create ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa_chain
