from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import chromadb
from config import CHROMA_PATH, COLLECTION_NAME
from ingest import chunk_text, ingest_pdf, load_pdf
from rag import SimpleRAG


TEST_CASES_PATH = Path(__file__).with_name("Testcase.json")
REPORT_PATH = Path(__file__).with_name("SystemTestReport.xlsx")
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def _xml_text(value: object) -> str:
    text = str(value)
    text = "".join(
        character
        for character in text
        if character in "\t\n\r"
        or "\u0020" <= character <= "\ud7ff"
        or "\ue000" <= character <= "\ufffd"
        or "\U00010000" <= character <= "\U0010ffff"
    )
    return escape(text[:30000])


def _column_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _worksheet_xml(rows: list[list[object]], widths: list[int]) -> str:
    row_xml = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(row, start=1):
            cell_ref = f"{_column_name(column_number)}{row_number}"
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
                f"{_xml_text(value)}</t></is></c>"
            )
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    columns = "".join(
        f'<col min="{number}" max="{number}" width="{width}" customWidth="1"/>'
        for number, width in enumerate(widths, start=1)
    )
    last_cell = f"{_column_name(len(widths))}{len(rows)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{columns}</cols><sheetData>{''.join(row_xml)}</sheetData>"
        f'<autoFilter ref="A1:{last_cell}"/>'
        "</worksheet>"
    )


def write_report(
    test_cases: list[dict],
    results: list[dict],
    started_at: str,
    run_status: str,
) -> None:
    passed = sum(result["status"] == "PASS" for result in results)
    failed = sum(result["status"] == "FAIL" for result in results)
    summary_rows = [
        ["Simple RAG system test report", ""],
        ["Status", run_status],
        ["Sample document", json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))["sample_document"]],
        ["Started (UTC)", started_at],
        ["Finished (UTC)", datetime.now(timezone.utc).isoformat()],
        ["Test cases", len(test_cases)],
        ["Passed", passed],
        ["Failed", failed],
    ]

    result_by_id = {result["id"]: result for result in results}
    result_rows = [[
        "Test ID",
        "Type",
        "Status",
        "Question",
        "Expected answer contains",
        "Actual answer",
        "Retrieved sources",
        "Duration (seconds)",
        "Details",
    ]]
    for case in test_cases:
        result = result_by_id.get(case["id"], {})
        result_rows.append([
            case["id"],
            case["type"],
            result.get("status", "PENDING"),
            case["question"],
            ", ".join(case["expected_answer_contains"]),
            result.get("answer", ""),
            ", ".join(result.get("sources", [])),
            result.get("duration_seconds", ""),
            result.get("details", ""),
        ])

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/>'
        '<sheet name="Test Results" sheetId="2" r:id="rId2"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(REPORT_PATH, "w", zipfile.ZIP_DEFLATED) as report:
        report.writestr("[Content_Types].xml", content_types)
        report.writestr("_rels/.rels", root_rels)
        report.writestr("xl/workbook.xml", workbook)
        report.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        report.writestr(
            "xl/worksheets/sheet1.xml",
            _worksheet_xml(summary_rows, [28, 60]),
        )
        report.writestr(
            "xl/worksheets/sheet2.xml",
            _worksheet_xml(
                result_rows,
                [28, 12, 12, 70, 45, 90, 35, 20, 50],
            ),
        )


def expected_chunks(pdf_path: Path) -> set[tuple[int, int]]:
    return {
        (page["page"], chunk_index)
        for page in load_pdf(str(pdf_path))
        for chunk_index, _ in enumerate(chunk_text(page["text"]))
    }


def ensure_sample_document_is_indexed(filename: str) -> None:
    pdf_path = DOCUMENTS_DIR / filename
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Required system-test PDF not found: {pdf_path}"
        )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    indexed = collection.get(
        where={"source": filename},
        include=["metadatas"],
    )
    indexed_chunks = {
        (metadata["page"], metadata["chunk"])
        for metadata in indexed["metadatas"] or []
    }
    required_chunks = expected_chunks(pdf_path)

    if indexed_chunks == required_chunks:
        print(
            f"Sample document is already indexed "
            f"({len(required_chunks)} chunks): {filename}"
        )
        return

    if indexed_chunks:
        print(f"Refreshing incomplete sample index: {filename}")
        collection.delete(where={"source": filename})
    else:
        print(f"Sample document is not indexed; ingesting: {filename}")

    count = ingest_pdf(
        str(pdf_path),
        collection,
        strategy="fixed_size",
        confirm_strategy=False,
    )
    if count != len(required_chunks):
        raise RuntimeError(
            f"Indexed {count} chunks, expected {len(required_chunks)} "
            f"for {filename}."
        )
    print(f"Indexed {count} sample chunks: {filename}")


def main() -> int:
    with TEST_CASES_PATH.open(encoding="utf-8") as testcase_file:
        test_data = json.load(testcase_file)

    sample_document = test_data["sample_document"]
    tests = test_data["tests"]
    started_at = datetime.now(timezone.utc).isoformat()
    test_results = []
    write_report(tests, test_results, started_at, "RUNNING")

    try:
        ensure_sample_document_is_indexed(sample_document)
    except Exception as exc:
        for test_case in tests:
            test_results.append({
                "id": test_case["id"],
                "status": "FAIL",
                "details": f"Sample document setup failed: {type(exc).__name__}: {exc}",
            })
        write_report(tests, test_results, started_at, "FAIL")
        print(f"Sample document setup failed: {type(exc).__name__}: {exc}")
        print(f"Report written to: {REPORT_PATH}")
        return 1

    rag = SimpleRAG()
    source_filter = {"source": sample_document}
    failures = 0

    for test_case in tests:
        test_id = test_case["id"]
        question = test_case["question"]
        test_started = time.perf_counter()
        sources = set()

        try:
            answer, rag_results = rag.ask(question, where=source_filter)
            sources = {
                metadata["source"]
                for metadata in rag_results["metadatas"][0]
            }
            expected_terms = test_case["expected_answer_contains"]
            missing_terms = [
                term
                for term in expected_terms
                if term.casefold() not in answer.casefold()
            ]

            passed = (
                bool(rag_results["documents"][0])
                and sources == {sample_document}
                and not missing_terms
            )
            detail = ""
            if not rag_results["documents"][0]:
                detail = "no matching sample-document chunks were retrieved"
            elif sources != {sample_document}:
                detail = f"unexpected retrieved sources: {sorted(sources)}"
            elif missing_terms:
                detail = f"answer missing expected text: {missing_terms}"
        except Exception as exc:
            passed = False
            answer = ""
            detail = f"{type(exc).__name__}: {exc}"

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {test_id} ({test_case['type']})")
        if answer:
            print(f"  Answer: {answer}")
        if detail:
            print(f"  Detail: {detail}")
        failures += not passed
        test_results.append({
            "id": test_id,
            "status": status,
            "answer": answer,
            "sources": sorted(sources),
            "duration_seconds": round(time.perf_counter() - test_started, 2),
            "details": detail,
        })
        write_report(
            tests,
            test_results,
            started_at,
            "RUNNING"
            if len(test_results) < len(tests)
            else ("FAIL" if failures else "PASS"),
        )

    passed_count = len(tests) - failures
    print(f"\nSystem test result: {passed_count}/{len(tests)} passed.")
    print(f"Excel report written to: {REPORT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
