# Simple RAG - 0$ RAG

Simple RAG is a local, PDF-based question-answering application. It uses
retrieval-augmented generation (RAG) to find relevant passages in indexed
documents and asks a locally running Ollama language model to answer using
those passages. The prompt instructs the model not to invent information and to
say when the documents do not contain an answer.

PDF text, embeddings, and the Chroma index are processed and stored locally.
The application requires Ollama and the configured models to be available on
the machine running it.

## How it works

1. **Ingest documents:** `ingest.py` profiles each PDF, recommends an available
   chunking strategy, and asks for confirmation or an alternative selection.
   After confirmation it chunks the document, generates embeddings with Ollama,
   and stores the chunks and page/section metadata in ChromaDB.
2. **Retrieve context:** `rag.py` delegates query rewriting, embeddings,
   Chroma search, deduplication, and parent-context expansion to `retriever.py`.
3. **Generate an answer:** `rag.py` delegates citation formatting, grounded
   prompt construction, and Ollama answer generation to `generator.py`.
4. **Chat:** `chat.py` provides an interactive prompt for asking questions.
5. **Profile documents:** `document_analyzer.py` extracts PDF text, reports page
   structure and layout indicators, and recommends a chunking approach before
   ingestion.
6. **Chunking strategies:** `chunking/` provides fixed-size, recursive,
   semantic, structure-aware, and parent-child chunkers with a shared metadata
   format.

## Project components

| Path | Purpose |
| --- | --- |
| `chat.py` | Interactive command-line interface. Reads questions, invokes `SimpleRAG`, prints answers, and exits on `exit` or `quit`. |
| `rag.py` | RAG orchestration and the `SimpleRAG.ask()` entry point. |
| `retriever.py` | Query embedding and rewriting, Chroma search, candidate filtering/deduplication, and parent-context expansion. |
| `generator.py` | Citation/source formatting, grounded prompt construction, and Ollama answer generation. |
| `rag_models.py` | Typed retrieval-result/chunk dataclasses and serialization to the legacy nested result shape for tracing and callers. |
| `ingest.py` | Profiles each PDF, prompts for the recommended or a chosen chunker, generates embeddings, and stores selected chunks with source, page-range, and section metadata in ChromaDB. |
| `document_analyzer.py` | Profiles one PDF, extracts page text, identifies likely headings, lists, table-like rows, and image-based pages, then writes a JSON report with a chunking recommendation. |
| `chunking/` | Shared chunk interface and five strategies. Chunk records contain content, inclusive page range, section path, index, and a nullable parent ID. |
| `config.py` | Runtime settings: Ollama model names, Chroma location and collection, chunk size/overlap, and retrieval count. |
| `tracing.py` | Shared console tracing for ingestion and RAG operations. Logs start/completion, elapsed seconds, Ollama token usage when reported, and exception details before re-raising failures. |
| `install.ps1` | Windows setup automation. Creates the Python environment, installs dependencies and machine prerequisites, downloads configured Ollama models, and indexes PDFs when needed. |
| `check_setup.py` | Post-install health check for Python, required configuration and folders, dependency versions, PDFs, Ollama service, and configured models. It does not inspect the generated Chroma index. |
| `requirements.txt` | Pinned Python dependencies used by the application. |
| `INSTALL.md` | Step-by-step Windows installation, health-check, system-test, document-ingestion, and usage instructions. |
| `Test/Testcase.json` | Positive and negative test questions and expected answer text for the sample RAG document. |
| `Test/systemtest.py` | Automated end-to-end test runner. Ensures only the sample PDF is indexed as needed, filters test retrieval to that sample, runs all test cases, and writes an Excel report. |
| `documents/` | Input PDF directory. Includes `sample_rag_test_4pages.pdf`, the fixture used by the system tests. |
| `chroma_db/` | Generated local Chroma index. Created at runtime; its contents are not source files and are excluded from Git. |

## Configuration

Edit `config.py` to change:

- `LLM_MODEL`: Ollama model used to generate answers (default: `qwen3:4b`).
- `EMBEDDING_MODEL`: Ollama model used for document and question embeddings (default: `nomic-embed-text`).
- `CHROMA_PATH` and `COLLECTION_NAME`: persistent index location and collection.
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: page-text chunking parameters.
- `TOP_K`: maximum number of matching chunks retrieved for a question.
- `MAX_RETRIEVAL_DISTANCE`: maximum Chroma distance for a result to be used;
  tune it against the distance metric and embeddings used by your collection.
- `LLM_NUM_CTX`: Ollama context-window size passed explicitly for generation.

Keep the configured Ollama models installed and available through the local
Ollama service.

RAG excludes parent records from similarity search and uses the selected child
records to fetch their parent text for answer context. It rejects results over
`MAX_RETRIEVAL_DISTANCE` and returns the refusal without calling the LLM if no
results remain. Prompt context includes document, page range, and section
metadata and instructs the model to cite its sources. Ollama generation uses the
explicit `LLM_NUM_CTX` value and temperature 0 rather than relying on server
defaults. Retrieved passages are treated as untrusted data. Retrieval
over-fetches candidates to fill `TOP_K` after duplicate chunks and repeated
parent contexts are removed. The chat client shows source locations and sends
recent conversation history for follow-up query reformulation.

Chunk IDs are deterministic for a document, selected strategy, and chunk
content. Re-ingesting the same source upserts existing records and removes
stale records for that source after successful indexing instead of multiplying
duplicate results.

## Quick start

On Windows, open PowerShell in the project directory and run:

```powershell
.\install.ps1
.\.venv-rag\Scripts\python.exe .\check_setup.py
.\.venv-rag\Scripts\python.exe .\Test\systemtest.py
.\.venv-rag\Scripts\python.exe .\chat.py
```

To profile a PDF before ingestion, run:

```powershell
.\.venv-rag\Scripts\python.exe .\document_analyzer.py .\documents\your-file.pdf
```

The report is written beside the PDF as `your-file_analysis.json`. To choose a
different report path, add `--output .\analysis\your-file.json`. The report
contains extracted text and page-level metrics, heuristic detections, and the
recommended chunking strategy. Scanned-page, heading, list, and table detection
are estimates; this tool does not perform OCR.

The chunkers accept pages as dictionaries with `page` and `text` keys and all
return a list of shared chunk dictionaries. For example:

```python
from chunking import FixedSizeChunker, StructureAwareChunker

pages = [{"page": 1, "text": "Section 1 Overview\nDocument text..."}]
fixed_chunks = FixedSizeChunker(chunk_size=800, overlap=100).chunk(pages)
section_chunks = StructureAwareChunker(chunk_size=800).chunk(pages)
```

`SemanticChunker` needs an embedding callable supplied by the caller (one text
string in, one numeric vector out), so it can use a selected embedding backend
without tying the package to a particular client. The parent-child strategy
returns parent and child records; children refer to their parent through
`parent_id` and use `chunk_type` to distinguish the records. These strategies
are available as a library and do not replace or change the current ingestion
pipeline. When using `document_analyzer.py`, pass its parsed JSON object to
`pages_from_analysis_report()` to preserve the report's detected heading lines.
The analyzer discovers the available options from each strategy module's
`Chunking strategy option` comment and includes their descriptions in its JSON
recommendation, so new strategies can be advertised by adding the same comment
format in their implementation file.

Ingestion analyzes each PDF first and saves `<document-name>_analysis.json`
beside it. Press Enter to accept the recommended strategy, enter another listed
strategy ID to choose it, or enter `q` to cancel ingestion. To batch accept
recommendations without prompting, run:

```powershell
.\.venv-rag\Scripts\python.exe .\ingest.py --yes
```

To explicitly select one strategy for every PDF, use:

```powershell
.\.venv-rag\Scripts\python.exe .\ingest.py --strategy recursive
```

The automated system test uses only `sample_rag_test_4pages.pdf` for retrieval,
even if other PDFs are present in `documents`. It generates
`Test/SystemTestReport.xlsx`, which is a local test artifact and is ignored by
Git.

For custom virtual-environment locations, manual setup commands, expected
prerequisites, and operational details, see [INSTALL.md](./INSTALL.md).

During ingestion and chat, operation traces are printed to the console. They
identify PDF extraction, page chunking, per-chunk embedding and Chroma writes,
query embedding, retrieval, prompt construction, and answer generation. A
failed operation logs its name, elapsed time, exception message, and traceback,
then the error is re-raised to the caller. Ollama embedding traces report input
tokens; follow-up query rewriting and answer-generation traces report
prompt/input tokens, generated/output tokens, and their total when the model
response supplies those counts. Other operations do not report token counts.
