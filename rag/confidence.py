"""
The 0-100 confidence score returned with every answer.

It describes how good the *retrieval* was (did we find plenty of closely matching text?), not whether the
LLM's wording is correct. It uses only the numbers already computed during search, so it costs nothing.

The score is built from four factors:

    similarity    up to 50 points   how close the chunks are to the question (the most important factor)
    count         up to 25 points   how many chunks were found (5 or more is full marks)
    consistency   up to 15 points   how similar the chunks' scores are to each other
    content       up to 10 points   whether there is enough text to answer from

A question that matches nothing (average similarity below 0.5) is also pulled down: the last three factors are
scaled to zero as the average similarity falls from 0.5 to 0.2. Without that, an off-topic question that still
returned eight long, evenly-scored chunks would look confident.
"""

from typing import List

from langchain_core.documents import Document


def calculate_confidence(docs: List[Document], similarity_scores: List[float] = None) -> float:
    """
    Score the retrieval from 0 (nothing useful) to 100 (strong, consistent, plentiful matches).

    Args:
        docs: the context chunks that will be shown to the LLM (used for the count and total length).
        similarity_scores: each chunk's semantic similarity (0 to 1), in the same order.
    Returns:
        The score rounded to one decimal place; 0.0 if there are no chunks.
    """
    if not docs:
        return 0.0

    num_docs = len(docs)

    # Factor 1: chunk count (0-25 points)
    if num_docs >= 5:
        count_score = 25.0
    elif num_docs >= 3:
        count_score = 20.0
    elif num_docs >= 1:
        count_score = 10.0
    else:
        count_score = 0.0

    # Factor 2: average similarity (10-50 points)
    if similarity_scores and len(similarity_scores) > 0:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        if avg_similarity >= 0.8:
            similarity_score = 50.0
        elif avg_similarity >= 0.7:
            similarity_score = 40.0
        elif avg_similarity >= 0.6:
            similarity_score = 30.0
        elif avg_similarity >= 0.5:
            similarity_score = 20.0
        else:
            similarity_score = 10.0

        # Factor 3: consistency (0-15 points). The standard deviation of the scores: the smaller, the better.
        if len(similarity_scores) > 1:
            score_std = (sum((s - avg_similarity) ** 2 for s in similarity_scores) / len(similarity_scores)) ** 0.5
            consistency_score = max(0, 15.0 * (1 - score_std * 2))
        else:
            consistency_score = 15.0
    else:
        # No scores available (should not happen): assume a middling result
        similarity_score = 25.0
        consistency_score = 7.5

    # Factor 4: content sufficiency (2.5-10 points), by total characters of context
    total_length = sum(len(doc.page_content) for doc in docs)
    if total_length >= 2000:
        content_score = 10.0
    elif total_length >= 1000:
        content_score = 7.5
    elif total_length >= 500:
        content_score = 5.0
    else:
        content_score = 2.5

    # Scale the count, consistency and content points down for poor matches (see the module docstring)
    if similarity_scores:
        relevance_gate = min(1.0, max(0.0, (avg_similarity - 0.2) / 0.3))
    else:
        relevance_gate = 1.0

    total_confidence = similarity_score + relevance_gate * (count_score + consistency_score + content_score)

    return round(total_confidence, 1)
