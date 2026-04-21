import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE", "./fastembed_cache")

# Text splitter — unchanged from the original tutorial.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Local embeddings via FastEmbed (ONNX, no torch dependency).
# - BAAI/bge-small-en-v1.5: 384-dim, ~130 MB, strong quality for its size.
# - Downloaded once to FASTEMBED_CACHE on first use, then cached forever.
# - No API key, no rate limits, no network calls after the initial download.
embedding_function = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir=FASTEMBED_CACHE,
)

# Persistent Chroma vector store.
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
)


def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


def index_document_to_chroma(file_path: str, file_id: int, user_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)

        # Tag every chunk with file_id AND user_id so we can:
        # - filter retrieval to only this user's chunks
        # - delete only this user's chunks on file deletion
        for split in splits:
            split.metadata["file_id"] = file_id
            split.metadata["user_id"] = user_id

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int, user_id: int) -> bool:
    """Delete chunks matching BOTH file_id and user_id — prevents one user
    from accidentally (or deliberately) wiping another's chunks."""
    try:
        vectorstore._collection.delete(
            where={"$and": [{"file_id": file_id}, {"user_id": user_id}]}
        )
        print(f"Deleted chunks for file_id={file_id} user_id={user_id}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id}: {e}")
        return False