"""
Finding the guidance passages that answer a question.

For each search query this module runs three stages:

    1. Hybrid search   two searches over the small "child" chunks: a semantic one (meaning, via embeddings) and a
                       keyword one (BM25, good for exact terms like "21 CFR 314.50" or "E14"). Their ranked
                       lists are merged with reciprocal rank fusion, which rewards chunks found by both.
    2. Rerank          optionally, a local cross-encoder model re-reads the top candidates against the question
                       and re-sorts them (only the order changes).
    3. Select          pick the final handful of chunks: at most MAX_CHUNKS_PER_DOC per guidance, never two
                       chunks from the same parent, and an even share across sub-queries.

Then `expand_to_parent_chunks` swaps each small child chunk for its larger parent chunk, which is what the LLM
actually reads. (Small chunks find things precisely; large chunks give the LLM enough context.)

Every chunk keeps its *semantic* similarity score (0 to 1, higher is better). The relevance check and the
confidence score are calibrated on that number, so fusion and reranking change the order but not the scores.
"""

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag import settings
from rag.clients import embeddings
from rag.store import DENSE_FIELD, HAS_SPARSE, OUTPUT_FIELDS, PK_FIELD, SPARSE_FIELD, milvus

# The optional cross-encoder is loaded once at startup. If it cannot be loaded the service still runs,
# just without reranking.
reranker = None
if settings.USE_RERANKING:
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
    except Exception as e:
        print(f"⚠️ Reranker failed to load, continuing without reranking: {e}")


def _hybrid_search(query: str, expr: Optional[str]) -> List[Tuple[Document, float]]:
    """
    Search for one query by meaning and by keywords, then merge and (optionally) rerank the results.

    Args:
        query: one standalone search text.
        expr: optional Milvus filter expression (see rag.filters), applied to both searches.
    Returns:
        (child chunk, semantic similarity) pairs, best first. A chunk found only by the keyword search has no
        semantic score of its own, so it gets the lowest semantic score among the candidates (its true value is
        at or below that).
    """
    search_args = dict(
        collection_name=settings.COLLECTION_NAME, limit=settings.FETCH_K, filter=expr or "", output_fields=OUTPUT_FIELDS
    )
    dense = milvus.search(
        data=[embeddings.embed_query(query)], anns_field=DENSE_FIELD,
        search_params=settings.DENSE_SEARCH_PARAMS, **search_args
    )[0]
    sparse = []
    if HAS_SPARSE:
        try:
            sparse = milvus.search(
                data=[query], anns_field=SPARSE_FIELD, search_params=settings.SPARSE_SEARCH_PARAMS, **search_args
            )[0]
        except Exception as e:
            print(f"Warning: keyword search failed, using semantic results only: {e}")

    # Reciprocal rank fusion: a chunk scores 1 / (RRF_K + rank) in each list it appears in, and the scores add.
    fused: Dict = {}
    entities: Dict = {}
    dense_scores: Dict = {}
    for hits, is_dense in ((dense, True), (sparse, False)):
        for rank, hit in enumerate(hits, start=1):
            pk = hit[PK_FIELD]
            fused[pk] = fused.get(pk, 0.0) + 1.0 / (settings.RRF_K + rank)
            entities[pk] = hit["entity"]
            if is_dense:
                dense_scores[pk] = hit["distance"]

    floor = min(dense_scores.values(), default=0.0)
    results = [
        (
            Document(
                page_content=entities[pk]["text"],
                metadata={k: v for k, v in entities[pk].items() if k != "text"},
            ),
            dense_scores.get(pk, floor),
        )
        for pk in sorted(fused, key=fused.get, reverse=True)
    ]
    return _rerank(query, results)


def _rerank(query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    Re-sort the best results by how well the cross-encoder thinks each chunk answers the query.

    Only the top RERANK_CANDIDATES are re-scored; the rest stay behind them in their old order. Each chunk keeps
    its semantic similarity score. If there is no reranker, or it fails, the input order is returned unchanged.
    """
    if reranker is None or len(results) < 2:
        return results
    head, tail = results[:settings.RERANK_CANDIDATES], results[settings.RERANK_CANDIDATES:]
    try:
        relevance = reranker.predict(
            [(query, doc.page_content) for doc, _ in head], batch_size=32, show_progress_bar=False
        )
    except Exception as e:
        print(f"Warning: reranking failed, using fused order: {e}")
        return results
    order = sorted(range(len(head)), key=lambda i: relevance[i], reverse=True)
    return [head[i] for i in order] + tail


def retrieve_documents(queries: List[str], expr: Optional[str] = None) -> Tuple[List[Document], List[float]]:
    """
    Run the full retrieval for one or more queries and return the chosen child chunks.

    Args:
        queries: the planner's standalone search texts (one for a simple question, up to 3 for a multi-part one).
        expr: optional Milvus filter expression applied to every query.
    Returns:
        (chunks, scores): up to TOP_K chunks for one query, or MULTI_QUERY_TOP_K for several, with each chunk's
        semantic similarity in the matching position of `scores`.
    """
    ranked = [_hybrid_search(q, expr) for q in queries]
    return _select_chunks(ranked, settings.TOP_K if len(queries) == 1 else settings.MULTI_QUERY_TOP_K)


def _select_chunks(ranked: List[List[Tuple[Document, float]]], limit: int) -> Tuple[List[Document], List[float]]:
    """
    Choose the final chunks from one ranked list per query.

    Takes the best chunk of every query, then the second best of every query, and so on, so that one part of a
    multi-part question cannot crowd out the others. A chunk is skipped if another chunk from the same parent was
    already chosen (they would show the LLM the same text) or if its guidance already has MAX_CHUNKS_PER_DOC chunks.
    Stops at `limit` chunks.
    """
    docs, scores = [], []
    seen_parents = set()
    per_source = {}

    for rank in range(max((len(r) for r in ranked), default=0)):
        for results in ranked:
            if rank >= len(results):
                continue
            doc, score = results[rank]
            parent_key = doc.metadata.get('parent_id') or doc.metadata.get('parent_chunk') or doc.page_content
            source = doc.metadata.get('source')

            if parent_key in seen_parents:
                continue
            if source and per_source.get(source, 0) >= settings.MAX_CHUNKS_PER_DOC:
                continue

            seen_parents.add(parent_key)
            per_source[source] = per_source.get(source, 0) + 1
            docs.append(doc)
            scores.append(score)

            if len(docs) == limit:
                return docs, scores

    return docs, scores


def expand_to_parent_chunks(docs: List[Document]) -> List[Document]:
    """
    Replace each small child chunk with the larger parent chunk it was cut from.

    The parent text (about 2,400 characters, starting with the document title, organization and date) is stored in
    the child's `parent_chunk` field. All other metadata is kept. A chunk without a parent is passed through as is.
    """
    expanded_docs = []

    for doc in docs:
        parent_chunk = doc.metadata.get('parent_chunk')

        if parent_chunk:
            expanded_docs.append(Document(page_content=parent_chunk, metadata=doc.metadata.copy()))
        else:
            expanded_docs.append(doc)

    return expanded_docs
