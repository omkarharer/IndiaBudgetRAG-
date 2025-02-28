import chainlit as cl
from src.helper import setup_qa_chain_pinecone_new,load_embedding,load_vector_db, load_llm, load_memory, setup_qa_chain, pinecone_setup, setup_qa_chain_pinecone, pinecone_setup_new

# Initialize components
pinecone = pinecone_setup()

# Step 1: Initialize Pinecone
# index = pinecone_setup_new()
embedding=load_embedding()
llm = load_llm()
memory = load_memory(llm)

# Set up QA chain
qa_chain = setup_qa_chain_pinecone(pinecone, llm, memory)
# Step 4: Set up QA chain
# qa_chain = setup_qa_chain_pinecone_new(index, embedding, llm, memory)

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    await cl.Message(content="🧞‍♂️ Welcome to BudgetBot! Your AI-powered financial analyst for the Indian Budget.").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""
    user_question = message.content
    
    # Send a placeholder message
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    try:
        # Get the response from the QA chain
        response = await cl.make_async(qa_chain)({"question": user_question})
        await cl.Message(content=response['answer']).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your question: {str(e)}").send()