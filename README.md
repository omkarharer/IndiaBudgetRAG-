
# IndiaBudgetRAG - Indian Budget Analyst

BudgetBot is an AI-powered financial analyst designed to provide insights and answers related to the Indian Budget. It leverages advanced natural language processing (NLP) and vector databases to deliver accurate and context-aware responses to user queries. The project includes two interfaces:

- Streamlit App: A user-friendly web interface for interacting with BudgetBot.

- Chainlit App: A conversational interface for seamless interaction with the AI.
The app uses two databases:

- Vector Database: For storing and retrieving embeddings of budget-related documents.

- Pinecone: A vector search engine for efficient similarity search and retrieval.

Checkout the demo using following URL=
https://omkarharer-financial-analyst-rag.streamlit.app/

## How It Works

User Input: The user enters a natural language question (e.g., "What is the budget allocation for income tax?").

Query Generation: The app uses a Language Learning Model (LLM) powered by Groq to generate a response based on the user's question.

Vector Search: The app queries the Pinecone vector database to retrieve relevant documents.

Response Generation: The app formats the retrieved data into a natural language response and displays it to the user.


```bash

       +-------------------+
       |   User Interface  |   <-- User Input (Natural Language)
       +-------------------+
               |
               v
       +---------------------------+
       |   LLM (Groq)             |   <-- Generates Response
       +---------------------------+
               |
               v
       +--------------------------+
       |  Pinecone Vector DB      |   <-- Retrieves Relevant Documents
       +--------------------------+
               |
               v
       +--------------------------+
       |   Response Generation    |   <-- Formats Data into Natural Language
       +--------------------------+
               |
               v
       +-------------------+
       |   User Interface  |   <-- Displays Answer
       +-------------------+
       
```

# 🚀 Steps to Run the IndiaBudgetRAG Locally

1️⃣ Clone the GitHub Repository

Open your terminal or command prompt and run:
   ```commandline
   git clone <repo-url>
   cd <your-repo-folder>
   ```



2️⃣ Create and Activate a Virtual Environment

Using Conda:
   ```commandline
   conda create -n IndiaBudgetRAG python=3.12 -y
conda activate IndiaBudgetRAG
   ```
Or using virtualenv (alternative to Conda):
   ```commandline
python -m venv IndiaBudgetRAG
source IndiaBudgetRAG/bin/activate  # On macOS/Linux
IndiaBudgetRAG\Scripts\activate     # On Windows

   ```

3️⃣ Install Required Dependencies

   ```commandline
pip install -r requirements.txt

   ```

4️⃣ Get API Key for LLM

- Obtain your API key from **groq console** using the link below:  
  - 🔗 [Get Your groq API Key](https://console.groq.com/keys)  
Create a .env file in the project directory and add your API key:

   ```commandline
GOOGLE_API_KEY=your_google_api_key_here
   ```

6️⃣ Run the Streamlit App
  ```commandline
streamlit run app.py
   ```


# 🚀Deployment to Streamlit Community Cloud

Push your code to GitHub.

Deploy the app to Streamlit Community Cloud.

Add your API key to Streamlit’s Secrets:
   ```commandline
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key
   ```
   Share the public URL (e.g., https://omkarharer-financial-analyst-rag.streamlit.app/) with others.

