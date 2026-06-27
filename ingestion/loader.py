"""
Document ingestion for the RAG pipeline.

Ensures the `documents/` folder exists and loads any PDFs found inside it
into LangChain Document objects ready for indexing.
"""

import os

from langchain_community.document_loaders import PyPDFLoader

DOCUMENTS_FOLDER = "documents"


def ensure_documents_folder(folder_name: str = DOCUMENTS_FOLDER) -> str:
    """Create the documents folder if it doesn't exist, and report status."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    return folder_name


def list_documents(folder_name: str = DOCUMENTS_FOLDER) -> list[str]:
    """List filenames currently inside the documents folder."""
    files = os.listdir(folder_name)

    if len(files) == 0:
        print(f"Folder '{folder_name}' doesn't contain any files.")
    else:
        print(f"Folder contains {len(files)} file(s).")
        for name in files:
            print(name)

    return files


def load_pdfs(folder_name: str = DOCUMENTS_FOLDER) -> list:
    """Load all PDF files in the documents folder into Document objects."""
    files = list_documents(folder_name)

    docs = []
    for file in files:
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_name, file)
            docs.extend(PyPDFLoader(pdf_path).load())

    print(f"Loaded {len(docs)} pages.")
    return docs
