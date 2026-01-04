from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def get_chunks(documents: List[Document]) -> List[Document]:
    """
    Takes a list of Documents (from PDF loader) and splits them into 
    manageable chunks while preserving metadata for source tracking.
    """
    
    if not documents:
        raise ValueError("Documents list cannot be empty")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        print(f"[*] Splitting {len(documents)} document pages into chunks...")

        chunks = text_splitter.split_documents(documents)

        # Azure Embeddings will fail if we send empty strings or pure whitespace
        valid_chunks = [
            chunk for chunk in chunks 
            if chunk.page_content and len(chunk.page_content.strip()) > 0
        ]
        filtered_count = len(chunks) - len(valid_chunks)
        if filtered_count > 0:
           print(f"Filtered out {filtered_count} invalid/empty chunks")
        
        print(f"[+] Created {len(valid_chunks)} valid chunks.")
        
        return valid_chunks
    
    except Exception as e:
        print(f"Error during document chunking: {str(e)}")
        raise