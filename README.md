## Agentic RAG: Domain-Knowledge QA with Self-Reflection
An advanced Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Azure OpenAI. Unlike standard RAG pipelines, this system utilizes an agentic loop with a self-reflection layer to validate retrieved context, minimize hallucinations, and provide grounded answers with source citations.

### Key Features
- Agentic Reasoning: Uses a "Plan-Act-Reflect" cycle to evaluate the relevance of retrieved documents before generating a response.

- Azure OpenAI Integration: Leverages enterprise-grade GPT-4.o-mini for reasoning and text-embedding-ada-002 for vectorization.

- Local Vector Store: Uses FAISS for high-performance, in-memory similarity search.

- Source Citation: Automatically tracks and cites the source filename (e.g., [policy_handbook.pdf]) for every claim.

- Robust Ingestion: Includes a recursive character splitter with data sanitization to prevent API errors from empty chunks.

## Setup & Installation
### Environment Configuration
Change example.env to .env :
```bash
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

AZURE_OPENAI_API_VERSION_CHAT=2025-01-01-preview
AZURE_OPENAI_API_VERSION_EMBEDDINGS=2023-05-15
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
## Prepare Data
Place your domain-specific PDF files in the ./data/ directory

## Usage 
Run the main chat loop to interact with your documents:
```bash
python app/main.py
```