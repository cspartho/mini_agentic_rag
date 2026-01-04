import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# ---- Constants ----
EMBEDDING_API_VERSION = "2023-05-15"
DEFAULT_TEMPERATURE = 0
SAFE_EMBEDDING_CHUNK_SIZE = 1  # safest for Azure debugging


# ---- Helpers ----
def _require_env(var_name: str) -> str:
    """Fetch required env var or raise a clear error."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value.strip()


# ---- Clients ----
def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """
    Returns an Azure OpenAI embedding client.
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=_require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=EMBEDDING_API_VERSION,
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        chunk_size=SAFE_EMBEDDING_CHUNK_SIZE,
    )


def get_azure_llm() -> AzureChatOpenAI:
    """
    Returns an Azure OpenAI chat LLM client.
    """
    return AzureChatOpenAI(
        azure_deployment=_require_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=_require_env("AZURE_OPENAI_API_VERSION_CHAT"),  # e.g. 2025-01-01-preview
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        temperature=DEFAULT_TEMPERATURE,
    )


# ---- Optional quick self-test ----
if __name__ == "__main__":
    llm = get_azure_llm()
    embeddings = get_azure_embeddings()

    print("LLM test:", llm.invoke("Say 'Azure LLM OK'").content)
    print("Embedding length:", len(embeddings.embed_query("Azure embeddings OK")))
