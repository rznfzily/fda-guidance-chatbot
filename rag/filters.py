"""
Filters a user can put on a search: FDA center, issue year range, and Final/Draft status.

Users never fill in a form. The planner (planner.py) reads them out of the question ("CDER final guidance since
2022") and hands the raw result to `clean_filters`, which keeps only values from the fixed lists below. Only
cleaned values are ever put into a Milvus filter expression, so text produced by the LLM cannot alter the query.

A "filters" dictionary looks like this (every key is optional):

    {"organizations": ["CDER", "CBER"], "year_from": 2022, "year_to": 2024, "status": "Final"}

This module has no network or database access, so it can be tested on its own.
"""

import datetime
from typing import Dict, Optional

# Filterable FDA organizations: short code -> the name stored in the `fda_organization` field.
# A guidance can list several organizations in that field, so matching uses LIKE, not equality.
ORGS = {
    "CDER": "Center for Drug Evaluation and Research",
    "CBER": "Center for Biologics Evaluation and Research",
    "CDRH": "Center for Devices and Radiological Health",
    "CVM": "Center for Veterinary Medicine",
    "CTP": "Center for Tobacco Products",
    "HFP": "Human Foods Program",
    "OCE": "Oncology Center of Excellence",
    "OII": "Office of Inspections and Investigations",
    "ORA": "Office of Regulatory Affairs",
    "OC": "Office of the Commissioner",
}
ORG_ALIASES = {"CFSAN": "HFP"}            # the former name of the Human Foods Program
STATUSES = {"final": "Final", "draft": "Draft"}  # lower-case input -> value stored in `guidance_status`
MIN_YEAR = 1990                            # earlier years are treated as a mistake and ignored


def clean_filters(raw) -> Dict:
    """
    Turn the planner's raw filter output into a safe filters dictionary.

    Unknown centers, years outside MIN_YEAR..next year, a start year after the end year, and any status other
    than Final/Draft are silently dropped.

    Args:
        raw: whatever the LLM returned for "filters" (normally a dict; anything else gives no filters).
    Returns:
        A dictionary with only the valid keys among organizations, year_from, year_to and status.
    """
    if not isinstance(raw, dict):
        return {}
    clean: Dict = {}

    orgs = []
    for code in raw.get("organizations") or []:
        code = ORG_ALIASES.get(str(code).strip().upper(), str(code).strip().upper())
        if code in ORGS and code not in orgs:
            orgs.append(code)
    if orgs:
        clean["organizations"] = orgs

    max_year = datetime.date.today().year + 1
    years = {}
    for key in ("year_from", "year_to"):
        try:
            year = int(raw.get(key))
        except (TypeError, ValueError):
            continue
        if MIN_YEAR <= year <= max_year:
            years[key] = year
    if len(years) == 2 and years["year_from"] > years["year_to"]:
        years = {}
    clean.update(years)

    status = STATUSES.get(str(raw.get("status") or "").strip().lower())
    if status:
        clean["status"] = status
    return clean


def build_filter_expr(filters: Dict, has_issue_year: bool) -> Optional[str]:
    """
    Build the Milvus filter expression for a cleaned filters dictionary.

    Example: {"organizations": ["CDER"], "year_from": 2022} becomes
    `(fda_organization like "%Center for Drug Evaluation and Research%") and issue_year >= 2022`.

    Args:
        filters: output of `clean_filters`.
        has_issue_year: whether the collection has the numeric `issue_year` field (store.HAS_ISSUE_YEAR). If not,
            years are matched against the end of the MM/DD/YYYY `issue_date` text instead.
    Returns:
        The expression, or None when there are no filters.
    """
    clauses = []
    if filters.get("organizations"):
        likes = [f'fda_organization like "%{ORGS[code]}%"' for code in filters["organizations"]]
        clauses.append("(" + " or ".join(likes) + ")")

    year_from, year_to = filters.get("year_from"), filters.get("year_to")
    if year_from or year_to:
        if has_issue_year:
            if year_from:
                clauses.append(f"issue_year >= {year_from}")
            if year_to:
                clauses.append(f"issue_year <= {year_to}")
        else:
            # issue_date is MM/DD/YYYY text, so match each year by suffix
            years = range(year_from or MIN_YEAR, (year_to or datetime.date.today().year) + 1)
            clauses.append("(" + " or ".join(f'issue_date like "%/{y}"' for y in years) + ")")

    if filters.get("status"):
        clauses.append(f'guidance_status == "{filters["status"]}"')
    return " and ".join(clauses) or None


def describe_filters(filters: Dict) -> str:
    """
    Describe the filters in a few words for the user, e.g. "CDER, since 2022, Final".

    This text is what the API returns as `filters` so a client can show which filters were applied.
    """
    parts = list(filters.get("organizations", []))
    year_from, year_to = filters.get("year_from"), filters.get("year_to")
    if year_from and year_to:
        parts.append(str(year_from) if year_from == year_to else f"{year_from}-{year_to}")
    elif year_from:
        parts.append(f"since {year_from}")
    elif year_to:
        parts.append(f"through {year_to}")
    if filters.get("status"):
        parts.append(filters["status"])
    return ", ".join(parts)
