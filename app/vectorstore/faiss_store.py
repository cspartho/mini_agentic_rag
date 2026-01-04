from llm.azure_configs import get_azure_embeddings
from langchain_community.vectorstores import FAISS

class FaissManager:
    def __init__(self):
        # This calls the Azure OpenAI embedding deployment we set up
        self.embeddings = get_azure_embeddings()

    def create_store(self, documents):
        """Creates and returns a FAISS vector store."""
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore