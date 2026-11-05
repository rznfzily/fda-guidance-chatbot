# FDA Guidance Chatbot

**RAG chatbot over FDA guidance documents: parent/child chunking, hybrid (semantic + BM25 keyword) search, an optional cross-encoder reranker, an LLM planner that routes questions and extracts filters, and streaming cited answers.**

> This is an independent personal project. It is not affiliated with or endorsed by the U.S. Food and Drug Administration, and its answers are not regulatory advice. Always check the cited FDA document.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How a Question Is Answered](#how-a-question-is-answered)
4. [Ingestion and Chunking](#ingestion-and-chunking)
5. [Confidence Scoring](#confidence-scoring)
6. [Failure Reporting](#failure-reporting)
7. [API Endpoints](#api-endpoints)
8. [Configuration](#configuration)
9. [Docker](#docker)
10. [Project Structure](#project-structure)
11. [Keeping the Data Up to Date](#keeping-the-data-up-to-date)
12. [Technology Stack](#technology-stack)
13. [Limitations and Ideas](#limitations-and-ideas)
14. [License](#license)

---

## Overview

The chatbot answers questions about FDA guidance documents using only their text. Documents are never stored on disk: PDFs are streamed from FDA URLs and read in memory, and web-page guidances are read from their FDA page.

### Key Features

- **Parent/child chunking** - small chunks (700 characters) are searched, the larger parent chunks (2,400 characters) they came from are sent to the LLM.
- **Hybrid search** - every chunk has a dense embedding and a BM25 keyword vector (computed by Milvus), merged by reciprocal rank fusion. Exact terms such as `21 CFR 314.50` or `E14` match as well as meaning does.
- **Reranking (optional)** - a local cross-encoder re-scores the merged candidates against the question.
- **Planner** - one LLM call per question routes small talk to a direct reply, rewrites follow-ups into standalone questions, splits multi-part questions into up to 3 searches, and extracts filters the user states (FDA center, issue year range, Final/Draft).
- **Relevance gate** - if nothing relevant is found the LLM is not called and the user is told the information isn't available.
- **Streaming and citations** - answers stream over SSE with numbered citations `[1]`, `[2]` and the source documents.
- **Confidence score** - a 0-100 retrieval-quality score returned with every answer.
- **Pluggable models** - OpenAI (or any OpenAI-compatible server), Azure OpenAI, or AWS Bedrock for the chat model.
- **Failure reports** - every ingestion run lists the documents that could not be processed, with a reason.

### Coverage

Figures from one full run (they change whenever FDA publishes guidance or you re-run the pipeline):

- **2,726 guidance records** in the scraped metadata (2,232 PDFs and 494 web pages; withdrawn guidances are skipped).
- **2,659 documents ingested** on the first full run, producing **210,797 chunks**. 67 failed because no text could be extracted; `--missing-only` recovers 10 of them (very short text), and the other 57 are image-only PDFs or empty pages that would need OCR.
- Documents are labelled with the issue date, FDA organization, topic and guidance status exactly as shown in the FDA table.

---

## Quick Start

### Prerequisites

- Python 3.12 (the Docker image uses the same version)
- Access to a chat model and an embedding model: an OpenAI API key, an Azure OpenAI resource, or AWS Bedrock (chat only; embeddings still come from OpenAI or Azure OpenAI)
- A Milvus 2.5+ server or Zilliz Cloud (Milvus 2.6 compatible) and `pymilvus` 2.6+
- Chrome/Chromium (for the Selenium scraper)

### Installation

```bash
git clone <this repository>
cd FDA_Guidance_Chatbot
python -m venv .venv && source .venv/bin/activate     # or: conda create -n chatbot python=3.12 && conda activate chatbot
pip install -r requirements.txt
cp .env.example .env                                  # then fill in your keys and Milvus connection
```

`.env` is git-ignored. Never commit it or bake it into an image. See [Configuration](#configuration) for every setting.

### Complete Workflow (in order)

**Step 1 - Scrape the FDA table**
```bash
python scripts/1_scrape_metadata_v2.py
```
Reads the FDA guidance table (headless Chrome) and writes `data/fda_guidance_metadata_v2.json` plus `data/scrape_failures_v2.csv`. Each record keeps its PDF link, or the FDA page link when the table has no document; withdrawn guidances are dropped. No document text is fetched here.

**Step 2 - Extract text and build chunks**
```bash
python scripts/2_ingest_pdfs.py
```
Streams each PDF (or reads each web page), cleans the text, builds parent/child chunks and writes `data/documents.pkl` plus the failure reports. It does **no** embedding and does not touch Milvus. Takes roughly 4-6 hours for the full set. Useful options:

- `--limit N` - ingest only N documents, evenly spaced across the list (for test runs; pair it with `--output` so the real pickle isn't replaced).
- `--missing-only` - ingest only documents not yet in the output pickle and append them to it (for example after fixing a failure).

Please be considerate of FDA's servers: the full run makes thousands of requests.

**Step 3 - Embed and upload to Milvus**
```bash
python scripts/3_upload_milvus.py
```
Embeds the chunks and uploads them in batches of 1,000. Milvus builds the BM25 vector itself during the upload, so one pass creates both indexes. **This drops and recreates the target collection.** `--collection NAME` overrides `MILVUS_COLLECTION`. Embedding the full set makes about 210,000 embedding calls' worth of text, so check your provider's pricing first.

**Step 4 - Start the API**
```bash
uvicorn fda_api_main:app --reload --host 0.0.0.0 --port 8000
```
Docs at http://localhost:8000/docs. The service refuses to start if the model credentials are missing or the collection does not exist. On Windows, `start_server.ps1` runs the same command from an activated environment.

---

## How a Question Is Answered

Everything below happens in the `rag/` package (entry point `rag/answer.py`), called from `fda_api_main.py`. Each file in `rag/` has a docstring explaining its role.

1. **Plan** (`plan_search`). One LLM call (`PLANNER_ALWAYS=true`, the default) returns a route, standalone queries and filters:
   - `direct`: greetings, thanks and clearly unrelated messages get a short reply with no search.
   - `search`: the question is rewritten using the chat history if needed, and multi-part questions become up to 3 standalone queries.
   - Filters are extracted only when the user states them ("CDER", "since 2022", "final guidance") and are checked against allow-lists before being used in the Milvus expression. There are no user-selected filters; request bodies that still send `filters`/`folder_filter` are accepted and ignored.
   - With `PLANNER_ALWAYS=false` the LLM is only called when a regex check or the chat history suggests it is needed. Any planner failure falls back to a plain search of the original question.
2. **Search** (`_hybrid_search`). Per query: embed it, run a semantic search and a BM25 keyword search (60 chunks each, with the filters), and merge the two lists by reciprocal rank fusion. Each chunk keeps its dense cosine score, which `MIN_RELEVANCE` and the confidence score are based on. If a filtered search finds nothing relevant, it is repeated without the filters.
3. **Rerank** (`_rerank`). If enabled, the cross-encoder re-scores the top 60 merged results and re-sorts them. Only the order changes.
4. **Select** (`_select_chunks`). Chunks are taken in turn across the queries; chunks from an already-chosen parent are skipped and at most 3 chunks per document are kept. The result is the top 8 chunks (12 when there are several queries).
5. **Expand** (`expand_to_parent_chunks`). Each child chunk is replaced by its parent chunk, which carries the title, organization and issue date.
6. **Relevance gate.** If the best semantic similarity is below `MIN_RELEVANCE` (0.35), the LLM is skipped and the standard "I don't have information about that" message is returned.
7. **Answer.** The numbered parent chunks go into the system prompt, which tells the LLM to answer only from them and cite `[n]`. The last 6 messages of chat history are included for understanding the question.
8. **Stream** (`fda_api_main.py`). The API sends the applied filters (if any), the tokens, the sources and the confidence over SSE.

With the default settings the planner and the answer are two separate LLM calls, so a searched question costs two (a direct reply costs one).

---

## Ingestion and Chunking

### Why parent/child?

Small chunks give precise search but too little context for the LLM; large chunks give context but noisy matches. The pipeline searches **child chunks** and sends their **parent chunks** to the LLM.

| | Child chunk | Parent chunk |
|---|---|---|
| Size / overlap | 700 / 100 characters | 2,400 / 300 characters |
| Used for | Search (dense + BM25) | LLM context |
| Header prepended | Title only | Title, FDA organization, issue date |

Children are cut from within a parent, so a child never spans two parents. Children under 100 characters (`MIN_CHUNK_CHARS`) are not indexed (their text stays in the parent). The constants are at the top of `scripts/2_ingest_pdfs.py`; changing any of them means re-running scripts 2 and 3.

The splitter is LangChain's `RecursiveCharacterTextSplitter` (paragraph, then line, then word breaks). It does **not** split by document headings, so a chunk can start or end mid-section.

### Text extraction

- **PDFs:** extracted with pypdf, then cleaned: running headers and footers (lines repeated on many pages), margin line numbers and bullet glyphs are removed.
- **Web pages:** the page body text is extracted from the FDA page, with retries and backoff on HTTP 403.
- **Image-only PDFs** have no text layer and cannot be ingested without OCR.

### Chunk metadata and IDs

Each chunk carries: `source` (the PDF or page link), `title`, `chunk_id`, `parent_id`, `chunk_index`, `parent_index`, `issue_date`, `issue_year` (int, 0 if unparseable), `source_type` (`pdf` or `html`), `fda_organization`, `topic`, `guidance_status`, `parent_chunk` and `source_label`. IDs are a hash of the source link, so documents with similar titles cannot collide. Collections built before `issue_year` and `source_type` existed still work (the year filter falls back to matching the date text).

### The Milvus collection

Created by `scripts/3_upload_milvus.py`: a dense `vector` (HNSW index, inner-product metric, search `ef=64`) and a `sparse` BM25 vector (English analyzer) computed by Milvus from `text`, plus the metadata fields above. The service queries it through a plain `pymilvus.MilvusClient` (pymilvus >= 2.6): the older LangChain Milvus wrapper cannot read a BM25 collection.

---

## Confidence Scoring

`calculate_confidence` returns 0-100 from four factors, using the **semantic (dense) similarity** of the selected chunks:

| Factor | Points | Rule |
|---|---|---|
| Average similarity | up to 50 | 50 if the average is at least 0.8, then 40 / 30 / 20 at 0.7 / 0.6 / 0.5, otherwise 10 |
| Chunk count | up to 25 | 25 for 5 or more, 20 for 3-4, 10 for 1-2 |
| Consistency | up to 15 | `15 x (1 - 2 x standard deviation)` of the scores, never below 0 |
| Content sufficiency | up to 10 | 10 for 2,000+ characters, 7.5 for 1,000+, 5 for 500+, otherwise 2.5 |

Off-topic questions still return full-looking chunks, so the last three factors are scaled down to zero as the average similarity falls from 0.5 to 0.2.

**Example:** scores `0.89, 0.87, 0.85, 0.83, 0.81` (average 0.85), five chunks, more than 2,000 characters: 50 + 25 + about 14 + 10 = about 99.

| Score | Interpretation |
|---|---|
| 90-100 | Strong match, consistent results |
| 75-89 | Good match |
| 60-74 | Acceptable, answer may need clarification |
| 40-59 | Weak match |
| 0-39 | Poor match, consider rephrasing |

These ranges are guidance for reading the number, not calibrated probabilities.

---

## Failure Reporting

After each ingestion run, the failures are written to `data/ingestion_failures.csv` and `.json`. The CSV columns are `title, fda_organization, issue_date, topic, url, guidance_status, error_type, error_detail`.

| Error type | Meaning | Likely cause / action |
|---|---|---|
| Invalid PDF (HTML page returned) | The link returned HTML, not a PDF | Withdrawn, moved or needs login; verify manually |
| No text extracted (empty or image-only) | Nothing could be read | Scanned PDF or empty page; needs OCR |
| PDF stream error | The PDF stream ended unexpectedly | Incomplete download; retry |
| Corrupt PDF (EOF marker missing) | Invalid PDF structure | Damaged file; retry or contact FDA |
| Network timeout | Connection timed out | Retry |
| Page load error | A web-page guidance could not be loaded | Often HTTP 403; retried with backoff, then reported |
| Page body not found | The page layout was not recognised | FDA page structure changed |
| Other error | Unclassified | Investigate manually |

Records the scraper itself could not use (for example a spreadsheet instead of a document) are listed in `data/scrape_failures_v2.csv`.

---

## API Endpoints

### Health
```
GET /api/health   ->   {"status": "healthy", "service": "FDA Guidance Chatbot"}
```
Use this path for Docker and load-balancer health checks.

### Non-streaming query
```
POST /api/query
{"question": "What are the requirements for bioequivalence studies?", "history": []}
```
Response:
```json
{
  "answer": "According to FDA guidance ... [1]",
  "sources": [
    {"content": "...", "source": "https://www.fda.gov/media/.../download",
     "metadata": {"title": "...", "issue_date": "08/03/2021", "issue_year": 2021,
                  "fda_organization": "Center for Drug Evaluation and Research",
                  "guidance_status": "Draft", "source_type": "pdf", "topic": "..."}}
  ],
  "confidence": 87.5,
  "filters": "CDER, since 2022"
}
```
`filters` is `null` unless the planner applied filters that returned relevant results.

### Streaming chat (SSE)
```
POST /api/chat
{"question": "CDER final guidance since 2022 about QT studies", "history": []}
```
Events, in order:
```
data: {"type": "filters", "description": "CDER, since 2022, Final"}   (only if filters were applied)
data: {"type": "token", "content": "An"}                              (repeated)
data: {"type": "sources", "sources": [...]}
data: {"type": "confidence", "score": 82.3}
data: {"type": "done"}
```
On failure the stream sends `{"type": "error", "message": ...}`. To watch tokens arrive live, use `curl -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" --data @body.json` (the Swagger page shows the whole stream only after it finishes).

### Follow-up questions
```
POST /api/followup
{"question": "...", "answer": "...", "sources": [...]}   ->   {"questions": ["...", "...", "..."]}
```

---

## Configuration

Settings come from environment variables, normally a `.env` file in the project root (loaded with `override=True`, so `.env` wins over the shell). `.env.example` lists them all. Only the model credentials and the Milvus connection are required.

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Chat model provider: `openai`, `azure` or `bedrock` |
| `LLM_MODEL` | `gpt-4.1-nano` | Model name (OpenAI), deployment name (Azure) or model / inference-profile id (Bedrock) |
| `EMBEDDING_PROVIDER` | same as `LLM_PROVIDER`, `openai` for Bedrock | `openai` or `azure` |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model or Azure deployment (3,072 dimensions). Must match what built the collection |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | - | OpenAI credentials; the base URL is optional (OpenAI-compatible servers) |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | - / - / `2024-10-21` | Azure OpenAI connection |
| `AWS_REGION` | `us-east-1` | Bedrock region; credentials come from the standard AWS chain (env vars, profile, IAM role) |
| `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_USER`, `MILVUS_PASSWORD`, `MILVUS_DB_NAME`, `MILVUS_SECURE` | `localhost`, `19530`, - , -, `default`, `false` | Milvus connection (`MILVUS_SECURE=true` means https) |
| `MILVUS_COLLECTION` | `fda_guidance_embeddings` | Collection to query / upload to |
| `PLANNER_ALWAYS` | `true` | `false` calls the planner LLM only when a regex check or history suggests it |
| `MAX_SUBQUERIES` | `3` | Most searches a multi-part question is split into |
| `MULTI_QUERY_TOP_K` | `12` | Chunks kept when there are several queries (one query keeps 8) |
| `MIN_RELEVANCE` | `0.35` | Best-chunk similarity below which no answer is attempted |
| `USE_RERANKING` | `false` | Enable the cross-encoder reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder id |
| `RERANK_CANDIDATES` | `60` | Candidates re-scored per query |
| `HF_HUB_OFFLINE` | - | Set to `1` once the reranker model is cached, so startup does not contact the Hugging Face Hub |

Behind a TLS-inspecting proxy, point the standard `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` variables at your CA bundle; the project does not ship or require one.

**Import order matters:** `rag/__init__.py` imports `rag/settings.py` first, which loads `.env`. `langchain_openai` imports `huggingface_hub`, which reads `HF_HUB_OFFLINE` only once at import, so do not import `langchain_openai` before `rag.settings`.

---

## Docker

The image runs the API only (as the non-root user `appuser`, port 8000). `data/` and `scripts/` are not included, and nothing secret is baked in.

### Build

```bash
docker build -t fda-chatbot .
```

On Apple Silicon use `docker buildx build --platform linux/amd64 -t fda-chatbot .` if you will deploy to x86_64 (the Fargate default).

The build installs CPU-only PyTorch (about 210 MB from `download.pytorch.org`, pinned by `TORCH_VERSION`), then `requirements.txt`, then downloads the reranker model from Hugging Face into the image and switches Hugging Face to offline mode, so the container needs no internet for the model at run time. The image is about 2.5 GB, and the first build takes roughly 15 minutes on a slow connection.

The `RERANKER_MODEL` build arg (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) names the baked-in model. The runtime `RERANKER_MODEL` must match it; if it does not, the container logs "Reranker failed to load" and runs without reranking. To bake a different model, pass `--build-arg RERANKER_MODEL=<model>` and use the same value at run time (or leave it unset at run time so the image's value is used).

### Requirements

`requirements.txt` is the only dependency file. It lists every package in the project: the web service and the `scripts/` extras (selenium, pypdf, ...). The image installs all of it even though the container never runs the scripts, so there is one file to maintain. Versions are not pinned (except `pymilvus`, which needs 2.6+, and PyTorch in the Dockerfile), so a build today may install newer releases than the author tested. If a future release breaks something, pin the offending package in `requirements.txt`. To freeze a known-good set, run `pip freeze` in a working environment.

### Run locally

```bash
docker run -d --name fda-chatbot -p 8000:8000 --env-file .env fda-chatbot
docker logs -f fda-chatbot
```

Startup takes a few minutes (Milvus connection, PyTorch import, reranker load); wait for `Application startup complete`. Then open http://localhost:8000/api/health and http://localhost:8000/docs. `docker ps` shows `(health: starting)` for up to 5 minutes, then `(healthy)`.

Notes:
- `--env-file` keeps quotes and trailing spaces that python-dotenv would trim. Keep values in `.env` free of them. (The app trims `HF_HUB_OFFLINE` itself.)
- Do not pass host file paths from `.env` (for example `SSL_CERT_FILE=C:\...`) into the container. Behind a TLS-inspecting proxy, mount your CA bundle and point both variables at the mounted path: `-v /path/ca.pem:/etc/ssl/custom-ca.pem:ro -e SSL_CERT_FILE=/etc/ssl/custom-ca.pem -e REQUESTS_CA_BUNDLE=/etc/ssl/custom-ca.pem`. On Windows Git Bash, prefix the command with `MSYS_NO_PATHCONV=1`.
- The container only queries Milvus, so restart it (not rebuild it) after loading more documents.

---

## Project Structure

```
FDA_Guidance_Chatbot/
├── fda_api_main.py              # FastAPI routes and request/response models
├── rag/                         # The question-answering engine (read in this order)
│   ├── settings.py              # Every setting and tunable number
│   ├── embeddings.py            # Embedding client (also used by the scripts)
│   ├── clients.py               # Chat model clients (OpenAI, Azure OpenAI, Bedrock)
│   ├── store.py                 # Milvus connection and collection layout
│   ├── filters.py               # FDA center / status / year filters
│   ├── planner.py               # Routes, rewrites and splits the question; extracts filters
│   ├── retrieval.py             # Hybrid search, reranker, chunk selection, parent expansion
│   ├── confidence.py            # The 0-100 confidence score
│   └── answer.py                # Prompt, answering, streaming, follow-up questions
├── scripts/
│   ├── 1_scrape_metadata_v2.py  # Scrape the FDA table
│   ├── 2_ingest_pdfs.py         # Extract text, build chunks -> data/documents.pkl
│   ├── 3_upload_milvus.py       # Embed and upload (dense + BM25); drops the target collection
│   └── update_fda.py            # Add guidances that are not yet in Milvus
├── data/                        # Pipeline output (git-ignored, not in the image)
│   ├── fda_guidance_metadata_v2.json, scrape_failures_v2.csv
│   ├── documents.pkl            # All chunks (input to script 3)
│   └── ingestion_failures.csv / .json
├── Dockerfile, .dockerignore
├── requirements.txt             # All Python packages (service and pipeline scripts)
├── .env.example                 # Settings template (copy to .env)
├── start_server.ps1             # Local dev server with auto-reload (Windows)
└── README.md
```

---

## Keeping the Data Up to Date

`python scripts/update_fda.py` adds guidances that are in the scraped metadata but not yet in Milvus, without dropping anything:

```bash
python scripts/update_fda.py --scrape      # refresh the metadata first, then add what is new
python scripts/update_fda.py --dry-run     # list what is new and stop
```

It ingests the new records with the same logic as script 2, embeds and appends them to the existing collection (Milvus builds the BM25 vectors), and appends the chunks to `documents.pkl` so a later full re-upload keeps them. Other options: `--limit N`, `--collection NAME`, `--docs PATH`.

It only detects **new** records. A guidance whose content changed under the same link, or one FDA has withdrawn, is not updated or removed. Rebuild with scripts 2 and 3 for that.

---

## Technology Stack

| Component | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| LLM | OpenAI, Azure OpenAI, or AWS Bedrock |
| Embeddings | `text-embedding-3-large` (3,072 dimensions) via OpenAI or Azure OpenAI |
| Vector database | Milvus 2.6 compatible: HNSW dense index and BM25 sparse index, `pymilvus` 2.6+ |
| Reranker | `sentence-transformers` cross-encoder (default MiniLM-L-6) |
| Text splitting / clients | LangChain (`RecursiveCharacterTextSplitter`, OpenAI clients) |
| PDF / web extraction | pypdf, requests |
| Scraping | Selenium and Chrome |

---

## Limitations and Ideas

- Test coverage is small and there is no retrieval evaluation set.
- Image-only PDFs (57 documents on the first run) are not ingested; OCR would be needed.
- Chunks do not follow document headings, so they carry no section name.
- Only new records are detected by `update_fda.py`.
- The API has no authentication.
- Planner and answer are separate LLM calls; the planner adds latency to every question by default.
- Ideas: OCR for scanned PDFs, section-aware chunking, a web UI, an evaluation set, a retrieval agent for list/count questions (a prototype was tried and removed).

---

## License

MIT, see [LICENSE](LICENSE). FDA guidance documents are U.S. government publications; this repository contains only code, not their content.
