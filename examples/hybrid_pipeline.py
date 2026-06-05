# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid retriever pipeline example.

Hybrid search combines dense vector similarity with BM25 keyword scoring in a
single TopK query.

Run:
    TOPK_API_KEY=<key> TOPK_REGION=<region> uv run python examples/hybrid_pipeline.py
"""

import os

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.topk import TopKHybridRetriever
from haystack_integrations.document_stores.topk import TopKDocumentStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

store = TopKDocumentStore(
    api_key=Secret.from_env_var("TOPK_API_KEY"),
    region=os.environ.get("TOPK_REGION", "aws-us-east-1-elastica"),
    collection_name="example-hybrid",
    embedding_dim=384,
    recreate_collection=True,
)

documents = [
    Document(content="The Python GIL prevents true thread-level parallelism; use multiprocessing to bypass it."),
    Document(
        content="A global interpreter lock serialises thread execution, so CPU-bound tasks don't benefit from threads."
    ),
    Document(
        content="OAuth 2.0 lets a user grant a third-party app access using tokens without sharing their password."
    ),
    Document(
        content="A protocol where users authorise external services to act on their behalf without revealing credentials."
    ),
    Document(
        content="A B-tree index speeds up reads but adds overhead on every write; consider partial indexes for write-heavy tables."
    ),
    Document(
        content="PostgreSQL B-tree and GIN indexes improve query speed at the cost of slower inserts and more disk space."
    ),
]

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")
indexing.run({"embedder": {"documents": documents}})

text_embedder = SentenceTransformersTextEmbedder(model=MODEL)
text_embedder.warm_up()

retriever = TopKHybridRetriever(document_store=store, top_k=2)
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

# Same intent expressed two ways — hybrid handles both
queries = [
    ("Python GIL multiprocessing workaround", "Python GIL multiprocessing"),
    ("my threads are not running in parallel, how do I fix it?", "threads parallel fix"),
    ("OAuth 2.0 token-based authorisation", "OAuth token authorisation"),
    ("let users log in with Google without giving us their password", "login google password"),
]

for text_query, keyword_query in queries:
    print(f"\nQuery: {text_query!r}")
    result = query_pipeline.run(
        {
            "embedder": {"text": text_query},
            "retriever": {"query": keyword_query},
        }
    )
    for doc in result["retriever"]["documents"]:
        print(f"  [{doc.score:.3f}] {doc.content}")
