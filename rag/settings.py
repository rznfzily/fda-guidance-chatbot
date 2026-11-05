"""
All settings and tunable numbers for the question-answering engine, in one place.

Values come from environment variables (normally the `.env` file in the project root, which is loaded here with
`override=True`, so `.env` wins over variables already set in the shell). Every setting has a default, so the
only ones you must provide are the model credentials and the Milvus connection. See README.md for what each one means.

Nothing here talks to the network; it only reads configuration.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# `docker run --env-file` and ECS keep stray spaces that python-dotenv would trim (HF_HUB_OFFLINE=1<space> became
# the value "1 ", which huggingface_hub does not recognise, so offline mode was silently off in the container).
# huggingface_hub reads this variable once, when it is first imported, so clean it here before anything imports it.
if "HF_HUB_OFFLINE" in os.environ:
    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_HUB_OFFLINE"].strip()


def _flag(name: str, default: str) -> bool:
    """Read an environment variable as a true/false switch ("true" in any capitalisation means True)."""
    return os.getenv(name, default).strip().lower() == "true"


# --- Language models -------------------------------------------------------------------------------------------

# Where the chat model runs: "openai" (OpenAI or any OpenAI-compatible server), "azure" (Azure OpenAI) or
# "bedrock" (Claude and other models on AWS Bedrock, using the standard AWS credential chain).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
if LLM_PROVIDER not in ("openai", "azure", "bedrock"):
    raise ValueError(f"LLM_PROVIDER must be openai, azure or bedrock (got '{LLM_PROVIDER}')")
USE_BEDROCK = LLM_PROVIDER == "bedrock"

# Chat model: an OpenAI model name, an Azure deployment name, or a Bedrock model / inference-profile id.
MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# Embeddings always come from OpenAI or Azure OpenAI (Bedrock is only used for chat). Defaults to the chat provider.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure" if LLM_PROVIDER == "azure" else "openai").strip().lower()
if EMBEDDING_PROVIDER not in ("openai", "azure"):
    raise ValueError(f"EMBEDDING_PROVIDER must be openai or azure (got '{EMBEDDING_PROVIDER}')")

# Embedding model (3,072-dimension vectors). Must match what scripts/3_upload_milvus.py used. For Azure this is
# the deployment name.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Azure OpenAI connection (used when a provider is "azure"; the key and endpoint are read from the environment).
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# AWS region for Bedrock.
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

# --- Milvus ----------------------------------------------------------------------------------------------------

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_SECURE = _flag("MILVUS_SECURE", "false")  # true = https
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "fda_guidance_embeddings")

# --- Search ----------------------------------------------------------------------------------------------------

FETCH_K = 60              # candidate chunks fetched per search (semantic and keyword each)
TOP_K = 8                 # chunks kept for a single-query question
MULTI_QUERY_TOP_K = int(os.getenv("MULTI_QUERY_TOP_K", "12"))  # chunks kept when the question was split
MAX_CHUNKS_PER_DOC = 3    # at most this many chunks from one guidance, so one document cannot fill the context
MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "3"))         # most searches a multi-part question becomes
RRF_K = 60                # reciprocal rank fusion constant (the standard value; higher = flatter weighting)

# Below this semantic similarity for the best chunk, nothing relevant was found: skip the LLM and say so.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.35"))

# Milvus search parameters. IP = inner product (equals cosine similarity for normalised embeddings).
DENSE_SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 64}}
SPARSE_SEARCH_PARAMS = {"metric_type": "BM25"}

# --- Reranker (optional local cross-encoder) -------------------------------------------------------------------

USE_RERANKING = _flag("USE_RERANKING", "false")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "60"))  # how many fused results get re-scored

# --- Planner ---------------------------------------------------------------------------------------------------

# true: the planner LLM sees every question. false: only questions that look like they need it (see planner.py).
PLANNER_ALWAYS = _flag("PLANNER_ALWAYS", "true")
