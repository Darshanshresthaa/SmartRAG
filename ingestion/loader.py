"""
Document ingestion for the RAG pipeline.

Ensures the `documents/` folder exists and loads any PDFs found inside it
into LangChain Document objects ready for indexing.
"""

import os

from langchain_community.document_loaders import PyPDFLoader

DOCUMENTS_FOLDER = "documents"


def ensure_documents_folder(folder_name: str = DOCUMENTS_FOLDER) -> str:
    """Create the documents folder if it doesn't exist."""
    os.makedirs(folder_name, exist_ok=True)
    print(f"Documents folder: {folder_name}")
    return folder_name


def list_documents(folder_name: str = DOCUMENTS_FOLDER) -> list[str]:
    """Return all files inside the documents folder."""
    files = sorted(os.listdir(folder_name))

    if not files:
        print(f"No files found in '{folder_name}'.")
    else:
        print(f"Found {len(files)} file(s):")
        for file in files:
            print(f"  • {file}")

    return files


def load_pdfs(folder_name: str = DOCUMENTS_FOLDER) -> list:
    """
    Load every PDF inside the documents folder.

    - Skips corrupted PDFs
    - Continues loading remaining files
    - Prints detailed errors
    """

    files = list_documents(folder_name)

    docs = []
    loaded_files = 0
    failed_files = []

    for file in files:

        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_name, file)

        print(f"\nLoading: {file}")

        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()

            docs.extend(pdf_docs)

            loaded_files += 1

            print(f"✓ Loaded {len(pdf_docs)} pages")

        except Exception as e:
            failed_files.append(file)

            print(f"✗ Failed to load '{file}'")
            print(f"Reason: {type(e).__name__}: {e}")

    print("\n" + "=" * 50)
    print("Loading Summary")
    print("=" * 50)
    print(f"PDFs loaded successfully : {loaded_files}")
    print(f"PDFs failed             : {len(failed_files)}")
    print(f"Total pages loaded      : {len(docs)}")

    if failed_files:
        print("\nFailed PDFs:")
        for file in failed_files:
            print(f" - {file}")

    return docs