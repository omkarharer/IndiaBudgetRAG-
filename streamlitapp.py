import streamlit as st
from src.helper import setup_qa_chain_pinecone, load_embedding, load_llm, load_memory, pinecone_setup

# Set the page title (tab name)
st.set_page_config(page_title="IndiaBudget")

# Initialize components
pinecone = pinecone_setup()
# embedding = load_embedding()
llm = load_llm()
memory = load_memory(llm)

# Set up QA chain
qa_chain = setup_qa_chain_pinecone(pinecone, llm, memory)

# Streamlit UI
with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "🔑[Get an groq API key](https://console.groq.com/keys)"
    "💻[View the source code](https://github.com/omkarharer/IndiaBudgetRAG-)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🧞‍♂️ Your AI-powered financial analyst for the Indian Budget.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get response from the QA chain
    try:
        response = qa_chain({"question": prompt})
        assistant_response = response['answer']
    except Exception as e:
        assistant_response = f"Error processing your question: {str(e)}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").write(assistant_response)