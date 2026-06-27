"""
Vector store management for the RAG pipeline.

Handles building, loading, and persisting the Chroma vector database,
and exposes a retriever for use in the LangGraph pipeline.
"""

from pathlib import Path

from langchain_community.vectorstores import Chroma

from service_llm.model import get_embedding_model

DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "books"

emb_model = get_embedding_model()


def load_vectorstore() -> Chroma:
    """Load an existing Chroma vectorstore from disk.

    Raises:
        FileNotFoundError: if the vectorstore has not been built yet.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Vector database not found. Build it first with build_vectorstore()."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb_model,
        persist_directory=str(DB_PATH),
    )


def build_vectorstore(documents) -> Chroma:
    """Create (or reuse) a Chroma vectorstore and add documents if empty.

    Args:
        documents: list of langchain Document objects to index.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb_model,
        persist_directory=str(DB_PATH),
    )

    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(documents)

    return vectorstore


def get_retriever(k: int = 3):
    """Return a retriever over the persisted vectorstore."""
    return load_vectorstore().as_retriever(search_kwargs={"k": k})
