"""
Push chunks to Milvus with simplified metadata schema

Strips down to only essential fields to avoid schema inference issues.
Each chunk gets a dense embedding ("vector") and a BM25 sparse vector ("sparse") that Milvus
computes itself from the chunk text, so the query service can run hybrid (semantic + keyword) search.
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import connections
from langchain_core.documents import Document

# The repo root must be importable when run as scripts/3_upload_milvus.py
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from rag.embeddings import make_embeddings

load_dotenv(override=True)

DOCS_PATH = REPO_ROOT / "data" / "documents.pkl"
BATCH_SIZE = 1000

def simplify_metadata(doc):
    """Keep only essential string/int fields, remove complex types"""
    simplified = Document(
        page_content=doc.page_content,
        metadata={
            "source": str(doc.metadata.get("source", "")),
            "title": str(doc.metadata.get("title", "")),
            "chunk_id": str(doc.metadata.get("chunk_id", "")),
            "parent_id": str(doc.metadata.get("parent_id", "")),
            "chunk_index": int(doc.metadata.get("chunk_index", 0)),
            "parent_index": int(doc.metadata.get("parent_index", 0)),
            "issue_date": str(doc.metadata.get("issue_date", "")),
            "issue_year": int(doc.metadata.get("issue_year", 0)),
            "source_type": str(doc.metadata.get("source_type", "")),
            "fda_organization": str(doc.metadata.get("fda_organization", "")),
            "topic": str(doc.metadata.get("topic", "")),
            "guidance_status": str(doc.metadata.get("guidance_status", "")),
            "parent_chunk": str(doc.metadata.get("parent_chunk", "")),
            "source_label": str(doc.metadata.get("source_label", "")),
        }
    )
    return simplified


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and upload them to Milvus (drops the target collection first)")
    parser.add_argument("--docs", type=Path, default=DOCS_PATH, help="Chunks pickle written by 2_ingest_pdfs.py")
    parser.add_argument("--collection", default=None,
                        help="Target collection; overrides MILVUS_COLLECTION from .env (use for test runs)")
    args = parser.parse_args()

    print("=" * 70)
    print("Push to Milvus (Simplified Schema)")
    print("=" * 70)
    print()

    # Load chunks
    if not args.docs.exists():
        print(f"❌ Chunks file not found: {args.docs}")
        return

    print(f"📂 Loading chunks from: {args.docs}")
    with open(args.docs, 'rb') as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded {len(chunks):,} chunks")
    print(f"   Simplifying metadata schema...")

    # Simplify all metadata
    simplified_chunks = [simplify_metadata(doc) for doc in chunks]

    print(f"   ✅ Metadata simplified (removed lists/arrays)")
    print(f"   Processing in batches of {BATCH_SIZE:,}")
    print()

    # Setup
    embeddings = make_embeddings()

    # Milvus settings
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
    MILVUS_SECURE = os.getenv("MILVUS_SECURE", "false").lower() == "true"
    COLLECTION_NAME = args.collection or os.getenv("MILVUS_COLLECTION", "fda_guidance_embeddings")

    connection_args = {
        "uri": f"{'https' if MILVUS_SECURE else 'http'}://{MILVUS_HOST}:{MILVUS_PORT}",
        "user": MILVUS_USER,
        "password": MILVUS_PASSWORD,
        "db_name": MILVUS_DB_NAME,
    }

    # Field order matters: dense first, BM25 sparse second (same order in index and search params)
    vector_fields = ["vector", "sparse"]
    bm25 = BM25BuiltInFunction(
        input_field_names="text",
        output_field_names="sparse",
        analyzer_params={"type": "english"},
    )
    index_params = [
        {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 128}},
        {"metric_type": "BM25", "index_type": "AUTOINDEX", "params": {}},
    ]
    search_params = [
        {"metric_type": "IP", "params": {"ef": 64}},
        {"metric_type": "BM25", "params": {}},
    ]

    print(f"🔄 Creating collection and uploading...")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Host: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   Database: {MILVUS_DB_NAME}")
    print()

    try:
        # First batch - creates collection
        first_batch = simplified_chunks[:BATCH_SIZE]
        print(f"📦 Batch 1/{(len(simplified_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}: Processing {len(first_batch):,} chunks...")

        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=COLLECTION_NAME,
            auto_id=True,
            vector_field=vector_fields,
            builtin_function=bm25,
            index_params=index_params,
            search_params=search_params,
            drop_old=True
        )
        # langchain-milvus reads the collection through the ORM API, but pymilvus 2.6 no longer
        # registers an ORM connection for a MilvusClient, so register it under the client's alias
        connections.connect(alias=vectorstore.alias, **connection_args)
        vectorstore.add_documents(first_batch)

        print(f"   ✅ Collection created and batch 1 uploaded")

        # Remaining batches
        total_batches = (len(simplified_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(1, total_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(simplified_chunks))
            batch = simplified_chunks[start_idx:end_idx]

            print(f"📦 Batch {i+1}/{total_batches}: {len(batch):,} chunks (total: {end_idx:,}/{len(simplified_chunks):,})...")

            vectorstore.add_documents(batch)

            print(f"   ✅ Batch {i+1} uploaded")

        print("\n" + "=" * 70)
        print("✅ Successfully pushed all chunks to Milvus!")
        print("=" * 70)
        print(f"Total chunks: {len(simplified_chunks):,}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Database: {MILVUS_DB_NAME}")
        print("\nNext: uvicorn fda_api_main:app --reload")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
