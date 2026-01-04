from ingestion.loader import load_pdfs_from_folder
from ingestion.chunker import get_chunks
from vectorstore.faiss_store import FaissManager
from tools.retrieval_tools import create_retriever_tool
from agent.agent import RAGAgent

def start_chat():
    print("--- SELISE AGENTIC RAG SYSTEM ---")
    
    raw_docs = load_pdfs_from_folder("./data")
    if not raw_docs:
        print("[-] No documents found. Please add PDFs to ./app/data and restart.")
        return

    chunks = get_chunks(raw_docs)
    
    fm = FaissManager()
    vector_db = fm.create_store(chunks)
    
    search_tool = create_retriever_tool(vector_db)
    agent_system = RAGAgent(tools=[search_tool])
    
    print("\n[!] System Ready. Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down. Goodbye!")
            break
        
        if not user_input:
            continue

        try:
            response = agent_system.run(user_input, search_tool)
            print(f"\nAI: {response}")
        except Exception as e:
            print(f"\n[ERROR]: {str(e)}")

if __name__ == "__main__":
    start_chat()