"""
The question-answering engine of the FDA Guidance Chatbot.

`fda_api_main.py` (the web API) calls into `rag.answer`; everything else lives here. The package is split so
that each file does one job. Reading them in this order follows a question from start to finish:

    settings    every setting and tunable number (read from .env)
    embeddings  the embedding client (also used by the ingestion scripts)
    clients     the chat model clients (OpenAI, Azure OpenAI or AWS Bedrock)
    store       the Milvus connection and the layout of the collection
    filters     FDA centers / status / year rules and the Milvus filter expression
    planner     one LLM call that routes the question, rewrites it, splits it and extracts filters
    retrieval   hybrid (meaning + keyword) search, optional reranker, picking the best chunks
    confidence  the 0-100 confidence score shown with each answer
    answer      builds the prompt, calls the LLM, streams the answer, writes follow-up questions

Importing `rag.settings` first (done below) loads the .env file before any other library is imported. That
order matters: `langchain_openai` pulls in `huggingface_hub`, which reads the HF_HUB_OFFLINE setting only once,
when it is first imported.
"""

from rag import settings  # noqa: F401  (loads .env; must stay first)
