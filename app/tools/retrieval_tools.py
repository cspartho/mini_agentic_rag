from langchain.tools import tool

def create_retriever_tool(vectorstore):
    @tool
    def search_docs(query: str) -> str:
        """Search the internal knowledge base and return content with sources."""
        docs = vectorstore.similarity_search(query, k=3)
        
        formatted_results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            formatted_results.append(f"Source [{source}]: {doc.page_content}")
            
        return "\n\n".join(formatted_results)
    return search_docs