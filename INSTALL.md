# Simple RAG installation and usage (Windows)

This guide assumes a Windows customer machine and PowerShell. The installer sets
up a dedicated Python 3.12 virtual environment, installs the pinned Python
dependencies, installs Ollama and its models if needed, and indexes PDFs already
placed in `documents`.

## 1. Install from the downloaded GitHub files

Open PowerShell in the folder containing `install.ps1`. If Windows blocks the
script, run it for this PowerShell session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Run the installer:

```powershell
.\install.ps1
```

The default environment is `.venv-rag`. To use a custom location:

```powershell
.\install.ps1 -VenvPath "C:\Users\YourName\Environments\simple-rag"
```

If Python 3.12 is installed at a non-standard location, provide its executable:

```powershell
.\install.ps1 -PythonPath "C:\Path\To\Python312\python.exe" `
  -VenvPath "C:\Users\YourName\Environments\simple-rag"
```

The installer uses `winget` to install missing Python 3.12, Ollama, and the
Visual C++ Build Tools required to build Chroma's Windows dependency. If Windows
requests a restart during Build Tools installation, restart and run the
installer again.

The installer downloads the models named in `config.py`. Model downloads require
an internet connection and several gigabytes of disk space. If PDFs are present
in `documents`, the installer indexes them unless an existing Chroma index is
already present.

## 2. Check the machine after installation

Run this check after the installer finishes. Use the Python executable from the
same virtual environment selected above:

```powershell
.\.venv-rag\Scripts\python.exe .\check_setup.py
```

For a custom environment path, substitute that path:

```powershell
& "C:\Users\YourName\Environments\simple-rag\Scripts\python.exe" .\check_setup.py
```

## 3. Run the automated RAG system test

Run the system test after the setup checker, using the same environment:

```powershell
.\.venv-rag\Scripts\python.exe .\Test\systemtest.py
```

For a custom environment:

```powershell
& "C:\Users\YourName\Environments\simple-rag\Scripts\python.exe" .\Test\systemtest.py
```

The checker verifies Python and package versions, checks that PDFs are
available, and checks the local Ollama service and both configured models. It
does not run a Chroma database smoke test or inspect `chroma_db`; the generated
index is created locally and is not included in Git. It exits with a non-zero
status and a diagnostic if a required check fails.

The test loads positive and negative cases from `Test\Testcase.json`. It
ensures `sample_rag_test_4pages.pdf` is indexed first; if the sample is missing
or incompletely indexed, it indexes or refreshes only that PDF. Test retrieval
is filtered to that sample document, so other PDFs in `documents` do not affect
the results. The run creates or updates `Test\SystemTestReport.xlsx` with a
summary and per-test answers/statuses. A non-zero exit code means at least one
test failed.

## 4. Add documents and rebuild the index

Copy text-based PDFs into `documents`. Then run ingestion using the selected
virtual environment:

```powershell
.\.venv-rag\Scripts\python.exe .\ingest.py
```

Ingestion adds chunks to the current index. Avoid rerunning it for the same PDFs,
because repeated runs can add duplicate chunks. To intentionally start over,
close the app and rename `chroma_db` to a backup name before running ingestion.

## 5. Run the application

Start the interactive question-and-answer program:

```powershell
.\.venv-rag\Scripts\python.exe .\chat.py
```

Ask a question at the prompt. Type `exit` or `quit` to close the program.

Ingestion and question answering print operation traces to the console,
including elapsed time for chunking, embeddings, Chroma writes, retrieval,
prompt construction, and answer generation. If an operation fails, the trace
includes the operation name, exception details, and traceback.
Ollama embedding traces include input-token counts; answer-generation traces
include input, output, and total token counts when provided by the model.

## Custom environment command reference

All Python dependencies are pinned in the repository-root `requirements.txt`.
To create and install into a virtual environment manually:

```powershell
py -3.12 -m venv .venv-rag
.\.venv-rag\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rag\Scripts\python.exe -m pip install -r .\requirements.txt
```

The installer automates the remaining machine prerequisites and model downloads.
