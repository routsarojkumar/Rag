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
from tracing import trace_operation


class SimpleRAG:

    def __init__(self):

        with trace_operation("Chroma client and collection initialization"):
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH
            )

            self.collection = self.client.get_collection(
                name=COLLECTION_NAME
            )

    def embed_query(self, query):

        with trace_operation("Query embedding") as trace:
            response = ollama.embed(
                model=EMBEDDING_MODEL,
                input=query
            )
            trace.record_ollama_usage(response)
            embedding = response["embeddings"][0]

        return embedding

    def retrieve(self, query, where=None):

        with trace_operation("Retrieval") as trace:
            query_embedding = self.embed_query(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                where=where
            )
            trace.record_retrieved_chunks(results)

            return results

    def build_prompt(self, query, results):

        with trace_operation("Prompt construction"):
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

        with trace_operation("Answer generation") as trace:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            trace.record_ollama_usage(response)

            return response["message"]["content"]

    def ask(self, query, where=None):

        with trace_operation("RAG question"):
            results = self.retrieve(query, where=where)

            prompt = self.build_prompt(
                query,
                results
            )

            answer = self.generate_answer(prompt)

            return answer, results