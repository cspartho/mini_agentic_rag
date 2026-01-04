import sys
from llm.azure_configs import get_azure_embeddings
from langchain_community.vectorstores import FAISS

class FaissManager:
    def __init__(self):
        self.embeddings = get_azure_embeddings()

    def create_store(self, documents):
        """Creates a FAISS vector store with robust error handling."""
        print(f"[*] Attempting to embed {len(documents)} chunks...")
        
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("[+] FAISS store created successfully.")
            return vectorstore
            
        except Exception as e:
            print("\n--- [DIAGNOSTIC ERROR REPORT] ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                if not content:
                    print(f"CRITICAL: Chunk index {i} is EMPTY. Azure will reject this.")
                if len(content) > 8000:
                    print(f"WARNING: Chunk index {i} is very large ({len(content)} chars).")
            
            print("--- [END OF REPORT] ---\n")
            raise e