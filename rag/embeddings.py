"""
The embedding client, shared by the query service and the ingestion scripts.

Kept apart from `clients.py` so the scripts can embed text without also building the chat model.
Credentials are read from the environment by the OpenAI libraries: OPENAI_API_KEY (and optionally OPENAI_BASE_URL)
for "openai", AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for "azure".
"""

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from rag import settings


def make_embeddings():
    """Build the embedding client for `settings.EMBEDDING_PROVIDER` and `settings.EMBEDDING_MODEL`."""
    if settings.EMBEDDING_PROVIDER == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.EMBEDDING_MODEL,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
