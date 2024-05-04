import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

#create vector database
def create_vector_db():
    # Load the data
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    # Split the data
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    # Load the embeddings
    embeddings = OpenAIEmbeddings()
    # Create the vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    # Ingest the data
    vectorstore.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()