from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

directory_path = "data/"


def load_documents_from_directory(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

document=load_documents_from_directory(data=directory_path)


def create_batch(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(data)
    return documents

texts = create_batch(data=document)
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


get_embeddings = get_embeddings()

DB_FAISS_PATH = "vectorstore/db_faiss"

db= FAISS.from_documents(
    texts,
    embedding=get_embeddings)
db.save_local(DB_FAISS_PATH)
