import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def test_chat_model():
    print("Testing Azure OpenAI Chat Model...")

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION_CHAT"],
        temperature=0,
    )

    response = llm.invoke("Say 'Chat model is working' in one short sentence.")
    print("Chat response:", response.content)


def test_embedding_model():
    print("\nTesting Azure OpenAI Embeddings...")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION_EMBEDDINGS"],
    )

    vector = embeddings.embed_query("This is a test sentence.")
    print("Embedding vector length:", len(vector))
    print("First 5 embedding values:", vector[:5])


if __name__ == "__main__":
    load_dotenv()

    try:
        test_chat_model()
        test_embedding_model()
        print("\n✅ Azure OpenAI chat and embedding keys are working correctly.")
    except Exception as e:
        print("\n❌ Test failed:")
        raise e
