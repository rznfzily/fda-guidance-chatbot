"""
The Milvus connection and what the guidance collection looks like.

The collection is created by `scripts/3_upload_milvus.py`. Each row is one small "child" chunk of a guidance
document with:

    text        the chunk text (a title line plus up to 700 characters)
    vector      its semantic embedding (dense vector)
    sparse      its BM25 keyword vector, computed by Milvus from `text` (cannot be read back)
    ...plus metadata: title, source link, issue date/year, FDA organization, status, and the larger
    `parent_chunk` text that is sent to the LLM.

Importing this module connects to Milvus, loads the collection into memory, and reads its field list. It fails
at once, with a clear message, if the collection does not exist, so a misconfigured service never starts.
"""

from pymilvus import DataType, MilvusClient

from rag import settings

milvus = MilvusClient(
    uri=f"{'https' if settings.MILVUS_SECURE else 'http'}://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
    user=settings.MILVUS_USER,
    password=settings.MILVUS_PASSWORD,
    db_name=settings.MILVUS_DB_NAME,
)
if not milvus.has_collection(settings.COLLECTION_NAME):
    raise RuntimeError(
        f"Milvus collection '{settings.COLLECTION_NAME}' does not exist. Run scripts/3_upload_milvus.py first."
    )
milvus.load_collection(settings.COLLECTION_NAME)

DENSE_FIELD = "vector"
SPARSE_FIELD = "sparse"

_schema_fields = milvus.describe_collection(settings.COLLECTION_NAME)["fields"]

# Name of the primary-key field (used to recognise the same chunk in two search result lists).
PK_FIELD = next(f["name"] for f in _schema_fields if f.get("is_primary"))

# Fields to request with every search: everything except the primary key and the two vector fields.
# Milvus refuses to return raw sparse (BM25) data, so asking for it would make the search fail.
OUTPUT_FIELDS = [
    f["name"] for f in _schema_fields
    if not f.get("is_primary") and f["type"] not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)
]

# False for a collection built without the BM25 field: the service then uses semantic search only.
HAS_SPARSE = any(f["name"] == SPARSE_FIELD for f in _schema_fields)

# False for a collection ingested before the numeric `issue_year` field existed: year filters then fall back to
# matching the end of the `issue_date` text (see filters.build_filter_expr).
HAS_ISSUE_YEAR = "issue_year" in OUTPUT_FIELDS
