"""
FDA Guidance Streaming Ingestion Pipeline

Processes FDA guidance documents WITHOUT downloading PDFs to disk.
- Streams PDFs from URLs directly into memory and extracts their text
- Page-text documents (blank Document column) are read from their FDA Summary page
- Builds parent/child chunks and saves them to data/documents.pkl

Nothing is sent to Milvus here. Run 3_upload_milvus.py afterwards to embed and upload.
"""

import argparse
import hashlib
import html
import json
import pickle
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Tuple

# PDF processing
import requests
from pypdf import PdfReader

# LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths (metadata comes from scripts/1_scrape_metadata_v2.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = REPO_ROOT / "data" / "fda_guidance_metadata_v2.json"
DATA_DIR = REPO_ROOT / "data"

# Processing settings - Parent-Child Pattern
CHILD_CHUNK_SIZE = 700  # Embedded and searched
CHILD_CHUNK_OVERLAP = 100
PARENT_CHUNK_SIZE = 2400  # Sent to the LLM
PARENT_CHUNK_OVERLAP = 300
MIN_CHUNK_CHARS = 100  # Smaller child chunks are not indexed (their text stays in the parent)
BATCH_SIZE = 50  # Process in batches to manage memory


def load_guidance_metadata() -> List[Dict]:
    """Load guidance metadata from JSON file"""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            f"Run 'python scripts/1_scrape_metadata_v2.py' first to generate it."
        )

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📚 Loaded {data['total_documents']} guidance documents from metadata")
    return data['documents']


def get_source_url(guidance: Dict) -> str:
    """PDF link for PDF documents; the FDA page URL for page-text documents (no pdf_url)."""
    return guidance.get('pdf_url') or guidance.get('landing_page_url') or ''


def stream_pdf_from_url(url: str, timeout=30) -> BytesIO:
    """Stream PDF content from URL into memory (no disk I/O)"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()

    return BytesIO(response.content)


def fetch_page_html(url: str, retries: int = 6) -> str:
    """Fetch an FDA page, backing off and retrying when the site refuses (e.g. HTTP 403)."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    last_error = "unknown error"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200:
                return response.text
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(3 * (attempt + 1))
    raise Exception(f"Could not load page ({last_error})")


def extract_text_from_page(page_html: str) -> str:
    """Body text of an FDA guidance page (title/date header, 'Issued by' box and comment-submission text excluded)"""
    m = re.search(r'<div class="col-md-8" role="main">(.*?)<!--BEGIN QUALTRICS', page_html, re.S)
    if not m:
        raise Exception("Page body not found")
    body = m.group(1)
    body = re.sub(r"<(script|style)\b.*?</\1>", "", body, flags=re.S | re.I)
    body = re.sub(r'<dl class="lcds-description-list--grid">.*?</dl>', "", body, flags=re.S)
    body = re.split(r"<h2>\s*Submit Comments", body, maxsplit=1)[0]
    body = re.sub(r"<br\s*/?>|</(p|div|li|h\d|tr|table|ul|ol)>", "\n", body, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


BULLET_RE = re.compile(r"^\s*[•●▪◦■□�]\s*")
TRAILING_NUMBER_RE = re.compile(r"^(.*?)(?:^|\s)(\d{1,3})\s*$")
EDGE_LINES = 3  # lines at the top/bottom of each page checked for running headers/footers


def _normalize_edge_line(line: str) -> str:
    """Key for spotting a header/footer repeated across pages (page numbers vary)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_line_numbers(lines: List[str]) -> List[str]:
    """Remove the margin line numbers of draft guidances (a consecutive 1..N run at line ends)."""
    matches = []
    for idx, line in enumerate(lines):
        m = TRAILING_NUMBER_RE.match(line)
        if m:
            matches.append((idx, m.group(1), int(m.group(2))))

    chain = []
    for pos, (idx, rest, num) in enumerate(matches):
        if pos and num == matches[pos - 1][2] + 1:
            chain.extend([matches[pos - 1][0], idx])
    if len(set(chain)) < 5:
        return lines

    strip = set(chain)
    return [TRAILING_NUMBER_RE.match(l).group(1) if i in strip else l for i, l in enumerate(lines)]


def clean_pdf_pages(pages: List[str]) -> str:
    """Join page texts, dropping running headers/footers, margin line numbers and bullet glyphs."""
    page_lines = [[l.rstrip() for l in p.splitlines()] for p in pages]

    edge_counts: Dict[str, int] = {}
    for lines in page_lines:
        non_blank = [l for l in lines if l.strip()]
        edges = {_normalize_edge_line(l) for l in non_blank[:EDGE_LINES] + non_blank[-EDGE_LINES:]}
        for key in edges:
            edge_counts[key] = edge_counts.get(key, 0) + 1
    min_pages = max(3, int(0.4 * len(pages)))
    repeated = {k for k, c in edge_counts.items() if k and c >= min_pages}

    cleaned_pages = []
    for lines in page_lines:
        kept = [l for l in lines if _normalize_edge_line(l) not in repeated]
        kept = _strip_line_numbers(kept)
        kept = [BULLET_RE.sub("- ", l) for l in kept]
        page_text = "\n".join(kept).replace("�", " ")
        page_text = re.sub(r"[ \t]+", " ", page_text)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        if page_text:
            cleaned_pages.append(page_text)
    return "\n\n".join(cleaned_pages)


def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    """Extract and clean text from in-memory PDF"""
    try:
        reader = PdfReader(pdf_bytes)
        pages = [page.extract_text() or "" for page in reader.pages]
        return clean_pdf_pages(pages)

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")


def create_embedding_header(guidance: Dict) -> str:
    """Title-only header for child chunks. Organization and date are identical across a document's
    chunks and would dominate a short chunk's vector, so they stay out of the embedded text."""
    return f"DOCUMENT: {guidance.get('title', 'Unknown Document')}\n---\n"


def create_context_header(guidance: Dict) -> str:
    """Full metadata header prepended to each parent chunk, which is what the LLM reads."""
    title = guidance.get('title', 'Unknown Document')
    fda_orgs = guidance.get('fda_organization', '')
    issue_date = guidance.get('issue_date', '')

    header = f"""DOCUMENT: {title}
FDA ORGANIZATION: {fda_orgs}
ISSUE DATE: {issue_date}
---
"""
    return header


def create_parent_child_chunks(text: str, guidance: Dict, min_chunk_chars: int = MIN_CHUNK_CHARS) -> List[Document]:
    """
    Create parent-child chunk pairs:
    - CHILD chunks (CHILD_CHUNK_SIZE chars): Indexed in Milvus for precise search
    - PARENT chunks (PARENT_CHUNK_SIZE chars): Stored in metadata for LLM context

    Each child is linked to its parent for context expansion during retrieval.
    """
    url = get_source_url(guidance)
    title = guidance['title']

    context_header = create_context_header(guidance)
    embedding_header = create_embedding_header(guidance)
    doc_key = hashlib.sha1((url or title).encode("utf-8")).hexdigest()[:12]

    # Display metadata, taken as-is from the FDA table
    issue_date = guidance.get('issue_date', '')
    fda_organization = guidance.get('fda_organization', '')
    topic = guidance.get('topic', '')
    # Numeric year (0 if unparseable) so the query service can filter on a range
    year_match = re.search(r"(\d{4})\s*$", issue_date or "")
    issue_year = int(year_match.group(1)) if year_match else 0
    source_type = guidance.get('source_type') or ('pdf' if guidance.get('pdf_url') else 'html')

    # Create PARENT splitter (large chunks)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    parent_chunks = parent_splitter.split_text(text)

    # Create CHILD splitter (small chunks)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    result_docs = []

    for parent_idx, parent_text in enumerate(parent_chunks):
        # Prepend context to parent
        parent_with_context = context_header + parent_text

        # Split parent into children, skipping fragments too short to search on
        child_texts = [c for c in child_splitter.split_text(parent_text) if len(c.strip()) >= min_chunk_chars]

        for child_idx, child_text in enumerate(child_texts):
            # Prepend the title to the child (this gets indexed)
            child_with_context = embedding_header + child_text

            # IDs keyed on the source URL so documents with similar titles cannot collide
            doc_id = f"{doc_key}_{parent_idx}_{child_idx}"
            parent_id = f"{doc_key}_{parent_idx}"

            # Create LangChain Document
            # IMPORTANT: page_content = CHILD (indexed for search)
            #            metadata['parent_chunk'] = PARENT (for LLM)
            doc = Document(
                page_content=child_with_context,
                metadata={
                    "source": url,
                    "title": title,

                    # Chunk identification
                    "chunk_id": doc_id,
                    "parent_id": parent_id,
                    "chunk_index": child_idx,
                    "parent_index": parent_idx,

                    # Display metadata (no filtering)
                    "issue_date": issue_date,
                    "issue_year": issue_year,
                    "source_type": source_type,
                    "fda_organization": fda_organization,
                    "topic": topic,
                    "guidance_status": guidance.get('guidance_status', ''),

                    # PARENT-CHILD PATTERN: Store parent chunk for LLM
                    "parent_chunk": parent_with_context,

                    # For display
                    "source_label": title,
                }
            )
            result_docs.append(doc)

    return result_docs


def process_guidance_document(guidance: Dict) -> List[Document]:
    """
    Process a single guidance document:
    1. Stream PDF from URL (no disk storage)
    2. Extract text from in-memory PDF
    3. Create parent-child chunks with context prepending
    4. Return LangChain Document objects with metadata

    Metadata (issue_date, fda_organization, topic, guidance_status) is display only, not used for filtering.

    Parent-Child Pattern:
    - Child chunks: Indexed for search
    - Parent chunks: Stored in metadata for LLM
    """

    title = guidance['title']

    try:
        if guidance.get('pdf_url'):
            # Stream PDF into memory and extract text
            pdf_bytes = stream_pdf_from_url(guidance['pdf_url'])
            text = extract_text_from_pdf(pdf_bytes)
        else:
            # Page-text document: read the body text of its FDA Summary page
            page_url = guidance.get('landing_page_url')
            if not page_url:
                raise ValueError("No pdf_url or landing_page_url")
            text = extract_text_from_page(fetch_page_html(page_url))

        if not text or not text.strip():
            raise ValueError("No text extracted (empty or image-only document)")

        # Create parent-child chunks with context prepending
        chunks = create_parent_child_chunks(text, guidance)
        if not chunks:
            # Very short document: every child is under MIN_CHUNK_CHARS, so keep them rather than lose the document
            chunks = create_parent_child_chunks(text, guidance, min_chunk_chars=1)

        return chunks

    except Exception as e:
        raise Exception(f"Error processing {title}: {e}")


def process_all_guidances(guidances: List[Dict]) -> tuple[List[Document], List[Dict]]:
    """
    Process all guidance documents and return chunks and failure report.

    Returns:
        tuple: (all_chunks, failed_documents_with_reasons)
    """
    all_chunks = []
    successful = 0
    failed = 0
    failed_docs = []  # Track failures with reasons

    print(f"\n🔄 Processing {len(guidances)} guidance documents...")
    print(f"   Parent-Child Pattern:")
    print(f"   - Child chunks: {CHILD_CHUNK_SIZE} chars (overlap: {CHILD_CHUNK_OVERLAP}) - indexed for search")
    print(f"   - Parent chunks: {PARENT_CHUNK_SIZE} chars (overlap: {PARENT_CHUNK_OVERLAP}) - stored for LLM")
    print(f"   - Context prepending: Enabled")
    print()

    for idx, guidance in enumerate(guidances, 1):
        title = guidance['title'][:60] + "..." if len(guidance['title']) > 60 else guidance['title']

        try:
            chunks = process_guidance_document(guidance)
            all_chunks.extend(chunks)
            successful += 1

            if idx % 10 == 0 or idx == len(guidances):
                print(f"   ✅ [{idx}/{len(guidances)}] {title} → {len(chunks)} chunks")

        except Exception as e:
            failed += 1
            error_msg = str(e)

            # Categorize error type
            if "invalid pdf header" in error_msg.lower():
                error_type = "Invalid PDF (HTML page returned)"
            elif "no text extracted" in error_msg.lower():
                error_type = "No text extracted (empty or image-only)"
            elif "stream has ended unexpectedly" in error_msg.lower():
                error_type = "PDF stream error"
            elif "eof marker not found" in error_msg.lower():
                error_type = "Corrupt PDF (EOF marker missing)"
            elif "timeout" in error_msg.lower():
                error_type = "Network timeout"
            elif "could not load page" in error_msg.lower():
                error_type = "Page load error"
            elif "page body not found" in error_msg.lower():
                error_type = "Page body not found"
            else:
                error_type = "Other error"

            # Store failure details
            failed_docs.append({
                'title': guidance['title'],
                'fda_organization': guidance.get('fda_organization', ''),
                'issue_date': guidance.get('issue_date', ''),
                'topic': guidance.get('topic', ''),
                'url': get_source_url(guidance),
                'guidance_status': guidance.get('guidance_status', ''),
                'error_type': error_type,
                'error_detail': error_msg[:200]  # Truncate long errors
            })

            print(f"   ❌ [{idx}/{len(guidances)}] {title} → FAILED: {error_type}")
            continue

    print(f"\n📊 Processing complete:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📄 Total chunks: {len(all_chunks)}")

    return all_chunks, failed_docs


def save_documents_for_bm25(chunks: List[Document], docs_path: Path = DATA_DIR / "documents.pkl"):
    """Save document chunks for BM25 hybrid search"""
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write to a temp file and swap it in, so a crash mid-write cannot corrupt an existing output
        tmp_path = docs_path.with_name(docs_path.name + ".tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(chunks, f)
        tmp_path.replace(docs_path)
        print(f"✅ Saved {len(chunks)} chunks for BM25: {docs_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save BM25 documents: {e}")


def save_failure_report(failed_docs: List[Dict]):
    """Save failure report to CSV"""
    import csv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "ingestion_failures.csv"
    json_path = DATA_DIR / "ingestion_failures.json"

    try:
        # Save CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if failed_docs:
                writer = csv.DictWriter(f, fieldnames=failed_docs[0].keys())
                writer.writeheader()
                writer.writerows(failed_docs)

        # Save JSON (more detailed)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_failed': len(failed_docs),
                'generated_at': datetime.now().isoformat(),
                'failures': failed_docs
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📋 Failure Report:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        print(f"   Total failures: {len(failed_docs)}")

        # Summary by error type
        from collections import Counter
        error_types = Counter(doc['error_type'] for doc in failed_docs)
        print(f"\n   Failures by type:")
        for error_type, count in error_types.most_common():
            print(f"   - {error_type}: {count}")

    except Exception as e:
        print(f"⚠️  Warning: Could not save failure report: {e}")


def get_topic_summary(chunks: List[Document]):
    """Generate summary of topics/categories"""
    topics = {}
    for chunk in chunks:
        topic = chunk.metadata.get('topic', 'General')
        topics[topic] = topics.get(topic, 0) + 1

    print(f"\n📚 Topic Distribution ({len(topics)} topics):")
    sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
    for topic, count in sorted_topics[:20]:
        print(f"   - {topic}: {count} chunks")
    if len(sorted_topics) > 20:
        print(f"   ... and {len(sorted_topics) - 20} more topics")


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk FDA guidance documents")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ingest only N documents, evenly spaced across the metadata list (for test runs)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "documents.pkl",
                        help="Where to write the chunks (use a different file for test runs)")
    parser.add_argument("--missing-only", action="store_true",
                        help="Ingest only documents whose source is not already in --output and append their chunks to it")
    args = parser.parse_args()

    print("=" * 70)
    print("FDA Guidance Streaming Ingestion Pipeline")
    print("=" * 70)
    print()

    try:
        # Load metadata
        guidances = load_guidance_metadata()
        if args.limit and args.limit < len(guidances):
            step = len(guidances) / args.limit
            guidances = [guidances[int(i * step)] for i in range(args.limit)]
            print(f"🧪 Test run: ingesting {len(guidances)} documents, evenly spaced across the list")

        existing_chunks = []
        if args.missing_only:
            with open(args.output, 'rb') as f:
                existing_chunks = pickle.load(f)
            done = {chunk.metadata.get('source') for chunk in existing_chunks}
            guidances = [g for g in guidances if get_source_url(g) not in done]
            print(f"♻️  Missing-only: {len(existing_chunks):,} chunks already in {args.output.name}, {len(guidances)} documents left to ingest")

        # Process all guidances (streaming, no disk storage)
        all_chunks, failed_docs = process_all_guidances(guidances)

        if not all_chunks:
            print("❌ No chunks created - ingestion failed")
            return
        all_chunks = existing_chunks + all_chunks

        # Show topic distribution
        get_topic_summary(all_chunks)

        # Save failure report
        if failed_docs:
            save_failure_report(failed_docs)

        # Save chunks; 3_upload_milvus.py embeds and uploads them
        save_documents_for_bm25(all_chunks, args.output)

        print("\n" + "=" * 70)
        print("✅ Ingestion Complete!")
        print("=" * 70)
        print(f"Total chunks: {len(all_chunks)}")
        print(f"\nNext step: Run 'python scripts/3_upload_milvus.py --docs {args.output}' to embed and upload to Milvus")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
