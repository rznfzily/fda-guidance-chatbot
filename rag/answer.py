"""
Answering a question: the entry point used by the web API (`fda_api_main.py`).

The three public functions are:

    answer_question(question, history)        returns the whole answer at once
    stream_answer(question, history)          yields the answer piece by piece as the LLM writes it
    generate_followup_questions(...)          suggests three follow-up questions

Both answering functions follow the same steps (all in `_prepare_context`, then the LLM call):

    1. plan        rag.planner decides: direct reply, or search with rewritten queries and any stated filters
    2. retrieve    rag.retrieval finds the best chunks (the filters are dropped if nothing relevant is found
                   under them) and swaps each for its larger parent chunk
    3. check       if even the best chunk is not relevant enough (settings.MIN_RELEVANCE), the LLM is not called
                   and the user gets NO_INFO_MESSAGE
    4. prompt      the numbered chunks go into SYSTEM_PROMPT, which tells the LLM to answer only from them and
                   to cite them as [1], [2], ...
    5. answer      the LLM (GPT through Azure OpenAI, or Claude through Bedrock) writes the answer

Importing this module prints a short startup summary of what is connected.
"""

from typing import Iterator, List, NamedTuple, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag import settings
from rag.clients import bedrock_client, complete, llm
from rag.confidence import calculate_confidence
from rag.filters import build_filter_expr, describe_filters
from rag.planner import plan_search
from rag.retrieval import expand_to_parent_chunks, reranker, retrieve_documents
from rag.store import HAS_ISSUE_YEAR, HAS_SPARSE

NO_INFO_MESSAGE = (
    "I don't have information about that in my FDA guidance knowledge base. "
    "Please try rephrasing your question or ask about a different topic covered in FDA guidance documents."
)

# Instructions given to the LLM with every answer. {context} receives the numbered guidance passages and
# {no_info} the exact sentence to use when the passages do not contain the answer.
SYSTEM_PROMPT = """
You are a knowledgeable assistant specialized in FDA regulatory guidance documents.

CRITICAL RULES:
1. You may ONLY answer questions using information found in the FDA guidance documents provided in the context below.
2. If the context is EMPTY or does not contain relevant information, you MUST respond EXACTLY with:
   "{no_info}"
3. Do NOT use your general knowledge or training data.
4. Do NOT make up information.
5. Do NOT provide answers about topics not covered in the FDA guidance documents.
6. When referencing guidance documents, mention their titles when available.
7. If multiple guidance documents are relevant, synthesize information from all of them.
8. Cite your sources. Each context block below is numbered like [1], [2]. After every claim taken from the context, add the number of the block it came from, e.g. "...three stages [2]." Use [1][3] when several blocks support a claim. Cite only numbers that exist below, and do not add a reference list at the end.
9. Earlier messages in the conversation are for understanding the question only. Citation numbers refer ONLY to the context blocks below, never to numbers used in earlier messages. Do not take facts from earlier answers unless the context below supports them.

Context from FDA Guidance Documents:
{context}

If the context above is empty or irrelevant, say you don't have information.
"""


class Prepared(NamedTuple):
    """Everything needed to answer one question, produced by `_prepare_context`.

    Attributes:
        docs: the context chunks (also returned to the user as the answer's sources).
        confidence: the 0-100 retrieval confidence score.
        system_prompt: the finished prompt for the LLM, or None when there is nothing to ask it (a direct reply,
            or no relevant passages were found).
        direct_reply: set for small talk / off-topic messages; the answer is this text and no search is made.
        filters: short text describing the filters that were applied (for example "CDER, 2023"), or None.
    """
    docs: List[Document]
    confidence: float
    system_prompt: Optional[str]
    direct_reply: Optional[str] = None
    filters: Optional[str] = None


def _build_prompt(parent_docs: List[Document]) -> str:
    """Number the context chunks [1], [2], ... and place them into SYSTEM_PROMPT."""
    context = "\n\n---\n\n".join([
        f"[{i}] Document: {doc.metadata.get('title', 'Untitled')}\n{doc.page_content}"
        for i, doc in enumerate(parent_docs, start=1)
    ])
    return SYSTEM_PROMPT.format(context=context, no_info=NO_INFO_MESSAGE)


def _prepare_context(question: str, history: List[dict] = None) -> Prepared:
    """
    Steps 1 to 4 of the module docstring: plan, retrieve, check relevance, build the prompt.

    If the user stated filters, the search runs with them first. If nothing relevant (similarity of at least
    MIN_RELEVANCE) comes back, or the filtered search fails, it is repeated without them, so a wrongly guessed
    filter can never hide the answer.

    Args:
        question: the user's latest message.
        history: earlier messages as [{"role": ..., "content": ...}].
    Returns:
        A `Prepared` result. `system_prompt` is None when the caller should reply without calling the LLM.
    """
    plan = plan_search(question, history)
    if plan.route == "direct":
        return Prepared([], 0.0, None, direct_reply=plan.reply)

    queries = plan.queries
    if len(queries) > 1:
        print(f"Sub-queries: {queries}")

    applied = None
    expr = build_filter_expr(plan.filters, HAS_ISSUE_YEAR)
    if expr:
        print(f"Filter: {expr}")
        try:
            child_docs, similarity_scores = retrieve_documents(queries, expr)
        except Exception as e:
            print(f"Warning: filtered search failed, searching without filters: {e}")
            child_docs, similarity_scores = [], []
        if similarity_scores and max(similarity_scores) >= settings.MIN_RELEVANCE:
            applied = describe_filters(plan.filters)
        else:
            print("Nothing relevant under the filter; searching without filters")
    if not applied:
        child_docs, similarity_scores = retrieve_documents(queries)

    weak = not similarity_scores or max(similarity_scores) < settings.MIN_RELEVANCE
    parent_docs = expand_to_parent_chunks(child_docs)
    confidence = calculate_confidence(parent_docs, similarity_scores)

    if weak:
        return Prepared([], confidence, None)

    return Prepared(parent_docs, confidence, _build_prompt(parent_docs), None, applied)


def _conversation(question: str, history: List[dict] = None) -> List[Tuple[str, str]]:
    """
    Combine the recent chat history and the new question into the turns sent to the LLM.

    Keeps the last six messages (long assistant replies are cut to 2,000 characters), merges consecutive messages
    from the same speaker, and makes sure the turns alternate user / assistant and start and end with "user",
    which Claude on Bedrock requires.

    Returns:
        A list of (role, text) pairs, where role is "user" or "assistant".
    """
    turns: List[Tuple[str, str]] = []
    for msg in (history or [])[-6:]:
        role = msg.get("role")
        text = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if role == "assistant":
            text = text[:2000]
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + text)
        else:
            turns.append((role, text))

    while turns and turns[0][0] != "user":
        turns.pop(0)

    if turns and turns[-1][0] == "user":
        turns[-1] = ("user", turns[-1][1] + "\n\n" + question)
    else:
        turns.append(("user", question))
    return turns


def _langchain_messages(system_prompt: str, turns: List[Tuple[str, str]]) -> list:
    """Convert the prompt and (role, text) turns into LangChain message objects for the GPT client."""
    messages = [SystemMessage(content=system_prompt)]
    for role, text in turns:
        messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
    return messages


def answer_question(
    question: str,
    history: List[dict] = None
) -> Tuple[str, List[Document], float, Optional[str]]:
    """
    Answer a question and return the finished answer (used by the non-streaming API endpoint).

    Args:
        question: the user's latest message.
        history: earlier messages as [{"role": "user" | "assistant", "content": "..."}].
    Returns:
        (answer, source_documents, confidence_score, applied_filters). For a direct reply or a "no information"
        answer the source list is empty. `applied_filters` is None when no filters were used.
    """
    parent_docs, confidence, system_prompt, direct_reply, filters = _prepare_context(question, history)
    if direct_reply:
        return direct_reply, [], 0.0, None
    if system_prompt is None:
        return NO_INFO_MESSAGE, [], confidence, None

    turns = _conversation(question, history)

    if settings.USE_BEDROCK:
        messages = [{"role": role, "content": [{"text": text}]} for role, text in turns]

        try:
            response = bedrock_client.converse(
                modelId=settings.MODEL,
                messages=messages,
                system=[{"text": system_prompt}],
                inferenceConfig={"maxTokens": 4096}
            )

            # The reply can contain several blocks (some are internal reasoning); take the first text block
            output = response.get('output', {})
            message = output.get('message', {})
            content = message.get('content', [])

            answer = ""
            for item in content:
                if 'text' in item:
                    answer = item['text']
                    break

            if not answer:
                answer = "Error: No text content in response"

        except Exception as e:
            print(f"Bedrock error: {e}")
            raise
    else:
        response = llm.invoke(_langchain_messages(system_prompt, turns))
        answer = response.content

    # Return the parent chunks: they are what the user sees as sources, and what the citations [1], [2] refer to
    return answer, parent_docs, confidence, filters


def stream_answer(
    question: str,
    history: List[dict] = None
) -> Iterator[Tuple[str, List[Document], float, Optional[str]]]:
    """
    Answer a question as a stream (used by the streaming API endpoint).

    Same inputs and steps as `answer_question`, but the answer arrives in small pieces as the LLM writes it.
    Nothing is yielded until the search is done, so the first piece may take a few seconds to appear.

    Yields:
        (token, source_documents, confidence_score, applied_filters) for every piece of text. The sources,
        confidence and filters are the same in every tuple, so the caller can send them once.
    """
    parent_docs, confidence, system_prompt, direct_reply, filters = _prepare_context(question, history)
    if direct_reply:
        yield direct_reply, [], 0.0, None
        return
    if system_prompt is None:
        yield NO_INFO_MESSAGE, [], confidence, None
        return

    turns = _conversation(question, history)

    if settings.USE_BEDROCK:
        messages = [{"role": role, "content": [{"text": text}]} for role, text in turns]

        try:
            response = bedrock_client.converse_stream(
                modelId=settings.MODEL,
                messages=messages,
                system=[{"text": system_prompt}],
                inferenceConfig={"maxTokens": 4096}
            )

            stream = response.get('stream')
            if stream:
                for event in stream:
                    if 'contentBlockDelta' in event:
                        delta = event['contentBlockDelta'].get('delta', {})
                        # Only text deltas are shown; internal reasoning deltas are skipped
                        if 'text' in delta:
                            token = delta['text']
                            if token:
                                yield token, parent_docs, confidence, filters
        except Exception as e:
            print(f"Bedrock streaming error: {e}")
            raise
    else:
        for chunk in llm.stream(_langchain_messages(system_prompt, turns)):
            token = chunk.content
            yield token, parent_docs, confidence, filters


def generate_followup_questions(
    question: str,
    answer: str,
    docs: List[Document]
) -> List[str]:
    """
    Suggest three follow-up questions the user might ask next.

    Args:
        question: the question that was asked.
        answer: the answer that was given.
        docs: the sources shown with the answer (accepted for the API's convenience; not used in the prompt).
    Returns:
        Up to three question strings.
    """
    prompt = f"""Based on this Q&A exchange about FDA guidance documents, generate 3 relevant follow-up questions:

Question: {question}
Answer: {answer}

Generate 3 follow-up questions that:
1. Dig deeper into the topic
2. Explore related FDA guidance areas
3. Clarify or expand on the answer

Return only the 3 questions, one per line, without numbering."""

    response = complete("You write follow-up questions about FDA guidance documents.", prompt, max_tokens=300)
    questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
    return questions[:3]


# Startup summary
print(f"✅ Connected to Milvus collection: {settings.COLLECTION_NAME}")
print(f"   Host: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
print(f"   Keyword search (BM25): {'on' if HAS_SPARSE else 'off'} | Reranking: {'on' if reranker else 'off'}")
