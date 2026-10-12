from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_PACKAGES = {
    "chromadb": "0.6.3",
    "ollama": "0.6.3",
    "pypdf": "6.20.0",
}


def report(label: str, ok: bool, detail: str = "") -> bool:
    state = "OK" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{state}] {label}{suffix}")
    return ok


def main() -> int:
    os.chdir(PROJECT_ROOT)
    failures = 0

    python_ok = sys.version_info[:2] == (3, 12)
    failures += not report(
        "Python version",
        python_ok,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        + ("" if python_ok else "; use the installed Python 3.12 virtual environment"),
    )

    required_config_files = ("config.py", "requirements.txt")
    for name in required_config_files:
        exists = (PROJECT_ROOT / name).is_file()
        failures += not report(f"Configuration file {name}", exists)

    required_folders = ("documents",)
    for name in required_folders:
        exists = (PROJECT_ROOT / name).is_dir()
        failures += not report(f"Required folder {name}", exists)

    packages_ok = True
    for package, expected_version in EXPECTED_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual_version = "not installed"
        package_ok = actual_version == expected_version
        packages_ok &= package_ok
        report(
            f"Python package {package}",
            package_ok,
            actual_version if package_ok else f"expected {expected_version}, found {actual_version}",
        )
    failures += not packages_ok

    try:
        import config
    except ImportError as exc:
        failures += not report("Application configuration", False, str(exc))
        print("\nSetup check failed. Rerun install.ps1 and then this checker.")
        return 1

    pdfs = list((PROJECT_ROOT / "documents").glob("*.pdf"))
    if pdfs:
        report("PDF documents", True, f"{len(pdfs)} found")
    else:
        report("PDF documents", False, "none found; add PDFs under documents before ingestion")
        failures += 1

    ollama_reachable = False
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags",
            timeout=5,
        ) as response:
            available_models = {
                model["name"]
                for model in json.load(response).get("models", [])
                if isinstance(model, dict) and isinstance(model.get("name"), str)
            }
        ollama_reachable = True
        report("Ollama service", True, "http://localhost:11434")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        available_models = set()
        failures += not report(
            "Ollama service",
            False,
            f"not reachable at http://localhost:11434 ({exc})",
        )

    for model_name in (config.EMBEDDING_MODEL, config.LLM_MODEL):
        installed = ollama_reachable and any(
            entry == model_name or entry.startswith(f"{model_name}:")
            for entry in available_models
        )
        detail = (
            ""
            if installed
            else "run install.ps1 to download it"
            if ollama_reachable
            else "cannot check until the Ollama service is available"
        )
        failures += not report(f"Ollama model {model_name}", installed, detail)

    if failures:
        print(f"\nSetup check failed: {failures} check(s) need attention.")
        return 1

    print(
        "\nAll checks passed. Start the application with: "
        f'"{sys.executable}" .\\chat.py'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
