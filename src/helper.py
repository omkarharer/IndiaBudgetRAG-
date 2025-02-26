import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

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