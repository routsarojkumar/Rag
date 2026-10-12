# Simple RAG

Simple RAG is a local, PDF-based question-answering application. It uses
retrieval-augmented generation (RAG) to find relevant passages in indexed
documents and asks a locally running Ollama language model to answer using
those passages. The prompt instructs the model not to invent information and to
say when the documents do not contain an answer.

PDF text, embeddings, and the Chroma index are processed and stored locally.
The application requires Ollama and the configured models to be available on
the machine running it.

## How it works

1. **Ingest documents:** `ingest.py` extracts text from PDFs in `documents`,
   splits each page into overlapping chunks, generates embeddings with Ollama,
   and stores each chunk and its source/page metadata in ChromaDB.
2. **Retrieve context:** for a user question, `rag.py` generates a query
   embedding and retrieves the most relevant chunks from the Chroma collection.
3. **Generate an answer:** `rag.py` adds the retrieved passages and question to
   a grounded prompt, then calls the configured Ollama language model.
4. **Chat:** `chat.py` provides an interactive prompt for asking questions.

## Project components

| Path | Purpose |
| --- | --- |
| `chat.py` | Interactive command-line interface. Reads questions, invokes `SimpleRAG`, prints answers, and exits on `exit` or `quit`. |
| `rag.py` | RAG implementation: query embedding, Chroma retrieval, context/prompt construction, answer generation, and the `SimpleRAG.ask()` entry point. |
| `ingest.py` | PDF extraction, text chunking, embedding generation, and persistent Chroma indexing. Its standard command indexes PDFs found in `documents`. |
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

Keep the configured Ollama models installed and available through the local
Ollama service.

## Quick start

On Windows, open PowerShell in the project directory and run:

```powershell
.\install.ps1
.\.venv-rag\Scripts\python.exe .\check_setup.py
.\.venv-rag\Scripts\python.exe .\Test\systemtest.py
.\.venv-rag\Scripts\python.exe .\chat.py
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
tokens; answer-generation traces report prompt/input tokens, generated/output
tokens, and their total when the model response supplies those counts. Other
operations do not report token counts.
