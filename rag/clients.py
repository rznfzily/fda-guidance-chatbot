"""
Connections to the language models: embeddings and the chat model.

Importing this module builds the clients, so the web API fails at startup if the credentials are missing instead
of on the first question. It exposes three things that the rest of the package imports:

    embeddings       turns a piece of text into a vector for semantic search
    llm              the LangChain chat client (OpenAI or Azure OpenAI), or None when Bedrock is configured
    bedrock_client   the AWS Bedrock runtime client, or None when OpenAI / Azure OpenAI is configured
    complete()       one plain, non-streaming call to whichever chat model is configured

Which chat model is used is decided by `settings.LLM_PROVIDER`.
"""

import boto3
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from rag import settings
from rag.embeddings import make_embeddings

embeddings = make_embeddings()

if settings.USE_BEDROCK:
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=settings.AWS_REGION)
    llm = None  # Bedrock is called through bedrock_client directly
else:
    bedrock_client = None
    if settings.LLM_PROVIDER == "azure":
        llm = AzureChatOpenAI(
            azure_deployment=settings.MODEL,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.0,
            streaming=True,
        )
    else:
        llm = ChatOpenAI(model=settings.MODEL, temperature=0.0, streaming=True)


def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Make one short, non-streaming call to the configured chat model.

    Used for the planner and for follow-up questions. The main answer does not use this because it streams.

    Args:
        system: instructions for the model (the "system prompt").
        user: the message to respond to.
        max_tokens: upper limit on the length of the reply.
    Returns:
        The model's reply as plain text (empty string if a Bedrock reply had no text block).
    """
    if settings.USE_BEDROCK:
        response = bedrock_client.converse(
            modelId=settings.MODEL,
            messages=[{"role": "user", "content": [{"text": user}]}],
            system=[{"text": system}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        for item in response.get('output', {}).get('message', {}).get('content', []):
            if 'text' in item:
                return item['text']
        return ""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content
