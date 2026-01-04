import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs_from_folder(folder_path: str):
    """Loads all PDFs and preserves filenames in metadata."""
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[*] Created {folder_path} folder. Drop your PDFs there.")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file
            documents.extend(loaded_docs)
    
    print(f"[+] Loaded {len(documents)} pages from {folder_path}")
    return documents