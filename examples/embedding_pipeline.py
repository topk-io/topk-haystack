# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

"""
Embedding retriever pipeline example.

Uses a local SentenceTransformers model to embed documents at index time and
queries at retrieval time. Unlike the semantic pipeline (which uses TopK built-in embedding),
this gives you full control over the embedding model — useful when you need a domain-specific model, a specific dimensionality,
or an embedding model you've fine-tuned yourself.

Documents and queries must be embedded with the same model.

Run:
    TOPK_API_KEY=<key> TOPK_REGION=<region> uv run python examples/embedding_pipeline.py
"""

import os

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.topk import TopKEmbeddingRetriever
from haystack_integrations.document_stores.topk import TopKDocumentStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

store = TopKDocumentStore(
    api_key=Secret.from_env_var("TOPK_API_KEY"),
    region=os.environ.get("TOPK_REGION", "aws-us-east-1-elastica"),
    collection_name="example-embedding",
    embedding_dim=384,
    recreate_collection=True,
)

# Same knowledge base as the semantic pipeline — programming languages described
# without naming them. Embeddings are computed locally by the SentenceTransformers model.
documents = [
    Document(
        content="Statically typed, compiles to machine code, ownership model guarantees memory safety without a garbage collector or runtime."
    ),
    Document(
        content="Interpreted, dynamically typed, famous for readable syntax and a vast ecosystem of scientific and data libraries."
    ),
    Document(
        content="Goroutines and channels are first-class primitives, making concurrent network services simple to write and reason about."
    ),
    Document(
        content="Compiles to bytecode, runs on a virtual machine, write-once-run-anywhere portability across operating systems."
    ),
    Document(
        content="Runs natively in the browser, dynamically typed, enables interactive interfaces that update without full page reloads."
    ),
    Document(
        content="A typed superset that adds static analysis and autocompletion to large JavaScript codebases without replacing the runtime."
    ),
    Document(
        content="A functional language with immutable data by default, pattern matching, and a type system that makes invalid states unrepresentable."
    ),
    Document(
        content="Designed for the JVM, combines object-oriented and functional styles, known for concise syntax and powerful collection APIs."
    ),
]

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")
indexing.run({"embedder": {"documents": documents}})

query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component("retriever", TopKEmbeddingRetriever(document_store=store, top_k=2))
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

queries = [
    "I want to add type checking to my existing web frontend without switching frameworks",
    "building a high-throughput API server that handles thousands of connections at once",
    "I need low-level performance but the program must never crash due to memory bugs",
    "I want the compiler to force me to handle every possible error and edge case",
]

for query in queries:
    print(f"\nQuery: {query!r}")
    result = query_pipeline.run({"embedder": {"text": query}})
    for doc in result["retriever"]["documents"]:
        print(f"  [{doc.score:.3f}] {doc.content}")
