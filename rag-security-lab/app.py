import os

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv(dotenv_path="../.env")

# --- 1. CONFIGURATION ---
# Configure the Gemini Pro model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Configure the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

print("Configuration loaded.")


# --- 2. INDEXING ---
def load_and_index_documents(docs_path="documents"):
    """
    Loads documents from a path, splits them, creates embeddings,
    and stores them in a Chroma vector database.
    """
    print("Loading documents from disk...")
    # We'll create the 'documents' directory and add files to it later
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    doc_files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
    if not doc_files:
        print("No .txt documents found. The app will run without a knowledge base.")
        # Create an empty placeholder file to avoid errors
        with open(os.path.join(docs_path, "placeholder.txt"), "w") as f:
            f.write("This is a placeholder. Add your own documents.")
        doc_files = ["placeholder.txt"]

    documents = []
    for doc_file in doc_files:
        loader = TextLoader(os.path.join(docs_path, doc_file), encoding="utf-8")
        documents.extend(loader.load())

    # Split documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    # Create the vector store (ChromaDB)
    print("Creating vector store with embeddings...")
    # This is the "indexing" step. It can take a moment.
    db = Chroma.from_documents(texts, embeddings)
    print("Vector store created.")
    return db


# Load documents and create the database when the app starts
db = load_and_index_documents()

# --- 3. QUERYING ---
# Create the RetrievalQA chain, which combines the retriever and the LLM
print("Creating QA chain...")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


def get_answer(question):
    """Runs the QA chain and returns the answer and sources."""
    print(f"New query received: {question}")
    result = qa_chain({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Format the source documents for display
    source_info = "\n\n**Sources:**\n"
    for doc in source_docs:
        # We access the 'source' metadata field from the document
        source_info += f"- {doc.metadata['source']}\n"

    return answer + source_info


print("QA chain ready.")

# --- 4. UI ---
# Create and launch the Gradio web interface
print("Launching Gradio interface...")
iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about your documents..."),
    outputs="text",
    title="Vulnerable RAG Chatbot",
    description="This chatbot is intentionally vulnerable. It answers questions based on the documents you provide.",
)

iface.launch()
print("Interface launched. Open the URL in your browser.")
