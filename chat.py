from rag import SimpleRAG


def main():
    rag = SimpleRAG()
    history = []

    print("Simple RAG - ask a question, or type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        answer, results = rag.ask(query, history=history)
        print(f"\nAnswer: {answer}")
        if results["sources"]:
            print("\nSources:")
            for source in results["sources"]:
                pages = (
                    f"page {source['page_start']}"
                    if source["page_start"] == source["page_end"]
                    else f"pages {source['page_start']}-{source['page_end']}"
                )
                section = (
                    f", section {' > '.join(source['section_path'])}"
                    if source["section_path"]
                    else ""
                )
                print(
                    f"- {source['source']} ({pages}{section}; "
                    f"distance {source['distance']:.3f})"
                )
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])


if __name__ == "__main__":
    main()
