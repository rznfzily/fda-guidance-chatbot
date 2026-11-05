"""
The planner: decides what to do with a question before any searching happens.

One LLM call reads the user's latest message (and the recent chat history) and returns a plan:

    route     "search" (look it up) or "direct" (a greeting / thanks / unrelated message answered at once)
    queries   1 to MAX_SUBQUERIES standalone search texts. Follow-ups like "and what about labeling?" are
              rewritten into full questions; a multi-part question is split so each part gets its own search.
    filters   FDA center, issue-year range and Final/Draft status, but only if the user stated them

Example: "What has CDER issued since 2022 about QT studies?" becomes
    queries=["QT/QTc interval prolongation studies"], filters={"organizations": ["CDER"], "year_from": 2022}

By default the planner runs on every question (settings.PLANNER_ALWAYS). If switched off, a cheap pattern check
decides whether a question needs it, and everything else goes straight to search. Any planner failure falls
back to a plain search of the original question, so planning can only help, never block an answer.
"""

import datetime
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from rag import settings
from rag.clients import complete
from rag.filters import ORGS, clean_filters

PLANNER_PROMPT = """You prepare a user's latest message for a search over FDA guidance documents.
Reply with ONE JSON object and nothing else:
{{"route": "search", "queries": ["..."], "filters": {{"organizations": [], "year_from": null, "year_to": null, "status": null}}, "reply": ""}}

route:
- "search" (default; use it whenever unsure): any question that FDA guidance documents might answer, including follow-ups.
- "direct": ONLY greetings, thanks, questions about what you can do, or messages clearly unrelated to FDA guidance (weather, jokes, coding, etc.). Put a one or two sentence reply in "reply" saying you answer questions about FDA guidance documents. Never answer factual questions in "reply".

queries (for "search"):
- Resolve pronouns and references ("it", "that guidance", "what about biologics?") using the conversation, if any.
- Keep specific terms: product types, guidance titles, regulations, section numbers. Leave out the filter wording (center, year, Final/Draft).
- One standalone query for a single topic (unchanged if already standalone). For a comparison or separate parts, one fully standalone query per part, at most {max_queries}.

filters: set ONLY what the user explicitly states, otherwise leave empty/null. Vague words like "recent" or "latest" set nothing. A follow-up keeps the filters from earlier turns unless the user changes them.
- organizations: codes from this list: {orgs}. (CFSAN means HFP.)
- year_from / year_to: integers. The current year is {year}. "in 2023" -> both 2023; "since 2022" -> year_from 2022; "before 2020" -> year_to 2019; "last 3 years" -> year_from {year}-3.
- status: "Final" or "Draft" only if the user says final or draft guidance."""

# Used only when PLANNER_ALWAYS is false: wording that suggests a question needs planning.
# Matches multi-part wording (compare, versus, "and also", two question marks), FDA center names, years,
# and words like since/after/before/final/draft.
PLANNER_TRIGGER_RE = re.compile(
    r"\b(compar\w*|differen\w*|differ|versus|vs\.?|between|both|as well as|and also)\b|;|\?.+\?|\band\b.{25,}"
    r"|\b(CDER|CBER|CDRH|CVM|CTP|OCE|OII|ORA|CFSAN|HFP|center|human foods|office of|since|after|before|final|draft)\b"
    r"|\b(19|20)\d{2}\b|\blast \d+ years?\b",
    re.IGNORECASE,
)
# Also only used when PLANNER_ALWAYS is false: greetings and "what can you do" style messages.
SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks?|thank you|good (morning|afternoon|evening)|who are you|what can you do|help)\b",
    re.IGNORECASE,
)


@dataclass
class SearchPlan:
    """What to do with a question.

    Attributes:
        route: "search" or "direct".
        queries: standalone search texts (always at least one for a "search" plan).
        filters: cleaned filters (see rag.filters), empty if the user stated none.
        reply: the answer to give straight away when route is "direct".
    """
    route: str = "search"
    queries: List[str] = field(default_factory=list)
    filters: Dict = field(default_factory=dict)
    reply: str = ""


def plan_search(question: str, history: List[dict] = None) -> SearchPlan:
    """
    Produce a plan for one question.

    Args:
        question: the user's latest message.
        history: earlier messages as [{"role": "user" | "assistant", "content": "..."}]. Only the last four are
            shown to the planner, which is enough to resolve "it" / "that guidance" references.
    Returns:
        A SearchPlan. If the planner is skipped (PLANNER_ALWAYS off and the question looks simple) or fails, the
        plan is a plain search of the original question with no filters.
    """
    plain = SearchPlan(queries=[question])
    if (not settings.PLANNER_ALWAYS and not history
            and not PLANNER_TRIGGER_RE.search(question) and not SMALLTALK_RE.search(question)):
        return plain

    if history:
        lines = []
        for msg in history[-4:]:
            content = msg.get("content", "")
            if msg.get("role") == "assistant":
                content = content[:500]
            lines.append(f"{msg.get('role', 'user')}: {content}")
        user_prompt = "Conversation:\n" + "\n".join(lines) + f"\n\nLatest message: {question}"
    else:
        user_prompt = f"Message: {question}"

    system = PLANNER_PROMPT.format(
        max_queries=settings.MAX_SUBQUERIES,
        year=datetime.date.today().year,
        orgs="; ".join(f"{code} = {name}" for code, name in ORGS.items()),
    )
    try:
        raw = complete(system, user_prompt, max_tokens=400)
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
    except Exception as e:
        print(f"Warning: query planning failed, using original question: {e}")
        return plain

    reply = str(data.get("reply") or "").strip()[:600]
    if data.get("route") == "direct" and reply:
        return SearchPlan(route="direct", reply=reply)

    queries, seen = [], set()
    for query in data.get("queries") or []:
        query = str(query).strip()
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)
    return SearchPlan(
        queries=queries[:settings.MAX_SUBQUERIES] or [question],
        filters=clean_filters(data.get("filters")),
    )
