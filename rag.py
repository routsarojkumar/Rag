# rag.py

import chromadb
import ollama

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)


class SimpleRAG:

    def __init__(self):

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def embed_query(self, query):

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query
        )

        return response["embeddings"][0]

    def retrieve(self, query, where=None):

        query_embedding = self.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            where=where
        )

        return results

    def build_prompt(self, query, results):

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []

        for i, (document, metadata) in enumerate(
            zip(documents, metadatas)
        ):

            source = metadata["source"]
            page = metadata["page"]

            context_parts.append(
                f"""
SOURCE {i + 1}
Document: {source}
Page: {page}

Content:
{document}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say:

"I don't have enough information in the provided documents."

Do not invent information.

Context:
-------------------------
{context}
-------------------------

User Question:
{query}

Answer:
"""

        return prompt

    def generate_answer(self, prompt):

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    def ask(self, query, where=None):

        results = self.retrieve(query, where=where)

        prompt = self.build_prompt(
            query,
            results
        )

        answer = self.generate_answer(prompt)

        return answer, results