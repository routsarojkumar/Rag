from rag import SimpleRAG


def main():
    rag = SimpleRAG()

    print("Simple RAG - ask a question, or type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        answer, _ = rag.ask(query)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
