"""
FDA Guidance Incremental Update

Adds guidances that are in the scraped metadata but not yet in Milvus, without dropping anything:
1. (optional, --scrape) re-run 1_scrape_metadata_v2.py to refresh data/fda_guidance_metadata_v2.json
2. list the `source` of every chunk already in the collection; a record is new if its source is missing
3. ingest the new records with the same logic as 2_ingest_pdfs.py (PDF or page text, parent/child chunks)
4. embed and append their chunks to Milvus (the BM25 vector is computed by Milvus on insert)
5. append the chunks to data/documents.pkl so a later full re-upload with 3_upload_milvus.py keeps them

Only new records are detected. A guidance whose content changed under the same link, or one FDA has since
withdrawn, is not updated or removed.

Usage:  python scripts/update_fda.py [--scrape] [--dry-run] [--limit N] [--collection NAME] [--docs PATH]
"""

import argparse
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, connections

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from rag.embeddings import make_embeddings

load_dotenv(override=True)

FLUSH_CHUNKS = 500  # flush to Milvus after whole documents once this many chunks are waiting


def load_script(name: str, filename: str):
    """Scripts 2 and 3 start with a digit, so they cannot be imported by name."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = load_script("ingest_pdfs", "2_ingest_pdfs.py")
upload = load_script("upload_milvus", "3_upload_milvus.py")


def indexed_sources(client: MilvusClient, collection: str) -> set:
    """Distinct `source` values (document links) of every chunk already in the collection."""
    pk = next(f["name"] for f in client.describe_collection(collection)["fields"] if f.get("is_primary"))
    sources = set()
    iterator = client.query_iterator(
        collection_name=collection, batch_size=5000, filter=f"{pk} >= 0", output_fields=["source"]
    )
    while True:
        rows = iterator.next()
        if not rows:
            iterator.close()
            break
        sources.update(row["source"] for row in rows)
    return sources


def main():
    parser = argparse.ArgumentParser(description="Add guidances that are not yet in Milvus")
    parser.add_argument("--scrape", action="store_true", help="Re-run 1_scrape_metadata_v2.py first")
    parser.add_argument("--dry-run", action="store_true", help="List the new guidances and stop")
    parser.add_argument("--limit", type=int, default=None, help="Ingest at most N new guidances (for test runs)")
    parser.add_argument("--collection", default=None,
                        help="Target collection; overrides MILVUS_COLLECTION from .env (use for test runs)")
    parser.add_argument("--docs", type=Path, default=upload.DOCS_PATH,
                        help="Chunks pickle to keep in sync (skipped if it does not exist)")
    args = parser.parse_args()

    print("=" * 70)
    print("FDA Guidance Incremental Update")
    print("=" * 70)

    if args.scrape:
        print("\n🔍 Refreshing metadata with 1_scrape_metadata_v2.py ...")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "1_scrape_metadata_v2.py")], check=True)

    guidances = ingest.load_guidance_metadata()

    host = os.getenv("MILVUS_HOST", "localhost")
    secure = os.getenv("MILVUS_SECURE", "false").lower() == "true"
    connection_args = {
        "uri": f"{'https' if secure else 'http'}://{host}:{os.getenv('MILVUS_PORT', '19530')}",
        "user": os.getenv("MILVUS_USER", ""),
        "password": os.getenv("MILVUS_PASSWORD", ""),
        "db_name": os.getenv("MILVUS_DB_NAME", "default"),
    }
    collection = args.collection or os.getenv("MILVUS_COLLECTION", "fda_guidance_embeddings")

    client = MilvusClient(**connection_args)
    if not client.has_collection(collection):
        print(f"\n❌ Collection '{collection}' does not exist. Run scripts/3_upload_milvus.py first.")
        return

    print(f"\n📋 Reading the documents already in '{collection}' ...")
    known = indexed_sources(client, collection)
    print(f"   {len(known):,} documents already indexed")

    new_guidances, seen = [], set()
    for guidance in guidances:
        url = ingest.get_source_url(guidance)
        if url and url not in known and url not in seen:
            seen.add(url)
            new_guidances.append(guidance)

    if not new_guidances:
        print("\n✅ No new guidances. The collection is up to date.")
        return
    print(f"\n🆕 {len(new_guidances)} new guidances")
    if args.limit:
        new_guidances = new_guidances[:args.limit]
        print(f"   Limited to the first {len(new_guidances)}")
    if args.dry_run:
        for guidance in new_guidances:
            print(f"   - {guidance['title'][:90]}")
        return

    embeddings = make_embeddings()

    # Same field layout as 3_upload_milvus.py. The collection and its indexes already exist.
    # langchain-milvus reads the collection through the ORM API, which pymilvus 2.6 no longer
    # registers for a MilvusClient, so register the connection under the client's alias first.
    connections.connect(alias=MilvusClient(**connection_args)._using, **connection_args)
    store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=collection,
        auto_id=True,
        vector_field=["vector", "sparse"],
        builtin_function=BM25BuiltInFunction(
            input_field_names="text", output_field_names="sparse", analyzer_params={"type": "english"}
        ),
        drop_old=False,
    )

    pending, added = [], []  # chunks waiting for Milvus / chunks Milvus has accepted
    failed = []

    def flush():
        if pending:
            store.add_documents([upload.simplify_metadata(chunk) for chunk in pending])
            added.extend(pending)
            pending.clear()

    print(f"\n🔄 Ingesting {len(new_guidances)} guidances ...\n")
    try:
        for idx, guidance in enumerate(new_guidances, 1):
            title = guidance["title"][:70]
            try:
                chunks = ingest.process_guidance_document(guidance)
            except Exception as e:
                failed.append((guidance["title"], str(e)[:150]))
                print(f"   ❌ [{idx}/{len(new_guidances)}] {title} → {str(e)[:100]}")
                continue
            pending.extend(chunks)
            print(f"   ✅ [{idx}/{len(new_guidances)}] {title} → {len(chunks)} chunks")
            # Whole documents only, so a failure never leaves a half-indexed document behind
            if len(pending) >= FLUSH_CHUNKS:
                flush()
        flush()
    finally:
        # Keep documents.pkl in step with whatever Milvus accepted, even if the run stopped early
        if added and args.docs.exists():
            with open(args.docs, "rb") as f:
                existing = pickle.load(f)
            ingest.save_documents_for_bm25(existing + added, args.docs)
        elif added:
            print(f"ℹ️  {args.docs} does not exist, so it was not updated")

    print("\n" + "=" * 70)
    print("✅ Incremental update complete")
    print("=" * 70)
    print(f"New chunks added to Milvus: {len(added):,}")
    print(f"Guidances that failed: {len(failed)}")
    for title, reason in failed[:15]:
        print(f"   - {title[:70]}: {reason}")
    if len(failed) > 15:
        print(f"   ... and {len(failed) - 15} more")


if __name__ == "__main__":
    main()
