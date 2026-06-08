# topk-haystack

Build RAG pipelines in a few lines of code with [TopK](https://topk.io) and [Haystack](https://haystack.deepset.ai/).

Ships with retrievers for every search mode — semantic (embeddings handled server-side, no embedder component needed),
dense vector, BM25 keyword, hybrid, and metadata filtering. Scales to billions of documents with native partition support for multi-tenant workloads.

## Installation

```bash
pip install topk-haystack
```

## Quick start

```python
import os
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.topk import TopKSemanticRetriever
from haystack_integrations.document_stores.topk import TopKDocumentStore

store = TopKDocumentStore(
    api_key=Secret.from_env_var("TOPK_API_KEY"), # Get your API key: https://console.topk.io/api-key
    region="aws-us-east-1-elastica", # See available regions: https://docs.topk.io/regions
    collection_name="my-docs",
)

# Index documents
indexing = Pipeline()
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.run({"writer": {"documents": [
    Document(content="Rust guarantees memory safety without a garbage collector."),
    Document(content="Python is known for readable syntax and scientific libraries."),
]}})

# Query
retriever = TopKSemanticRetriever(document_store=store, top_k=2)
pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
result = pipeline.run({"retriever": {"query": "memory safe systems programming"}})
for doc in result["retriever"]["documents"]:
    print(f"[{doc.score:.3f}] {doc.content}")
```

Set `TOPK_API_KEY` in your environment. Get your API key from the [TopK console](https://console.topk.io/api-key).

## Document store

```python
TopKDocumentStore(
    region="aws-us-east-1-elastica",   # required — see https://topk.io/docs/regions
    api_key=Secret.from_env_var("TOPK_API_KEY"),
    collection_name="haystack",        # collection to create or reuse
    embedding_dim=768,                 # vector dimension (must match your embedder)
    similarity="cosine",               # "cosine" | "euclidean" | "dot_product"
    recreate_collection=False,         # drop and recreate on init
    filter_documents_limit=10_000,     # cap for filter_documents()
    partition=None,                    # optional partition for multi-tenant use
)
```

TopK uses upsert semantics — documents with the same ID are overwritten when using `DuplicatePolicy.NONE` or `DuplicatePolicy.OVERWRITE`. `DuplicatePolicy.SKIP` and `DuplicatePolicy.FAIL` are not supported and raise a `ValueError`.

TopK can only return metadata fields that are explicitly selected. This integration automatically returns `meta.*` fields referenced in filters; unfiltered queries return documents without metadata.

## Retrievers

### Semantic (server-side embedding)

TopK embeds documents and queries server-side. No embedder component needed.

```python
from haystack_integrations.components.topk import TopKSemanticRetriever

retriever = TopKSemanticRetriever(document_store=store, top_k=5)
pipeline.add_component("retriever", retriever)
result = pipeline.run({"retriever": {"query": "your question here"}})
```

### Dense vector (bring your own embedder)

Embed documents and queries with your own model (e.g. `SentenceTransformers`).
`embedding_dim` in `TopKDocumentStore` must match the model's output dimension.

```python
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.topk import TopKEmbeddingRetriever

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Indexing — embed before writing
indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")

# Querying
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component("retriever", TopKEmbeddingRetriever(document_store=store, top_k=5))
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
result = query_pipeline.run({"embedder": {"text": "your question here"}})
```

### BM25 keyword

```python
from haystack_integrations.components.topk import TopKBM25Retriever

retriever = TopKBM25Retriever(document_store=store, top_k=5)
pipeline.add_component("retriever", retriever)
result = pipeline.run({"retriever": {"query": "keyword search terms"}})
```

### Hybrid (vector + BM25)

Combines dense vector similarity with BM25 keyword scoring in a single query. Takes both a text embedding and a keyword query string.

```python
from haystack_integrations.components.topk import TopKHybridRetriever

retriever = TopKHybridRetriever(document_store=store, top_k=5)
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component("retriever", retriever)
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
result = query_pipeline.run({
    "embedder": {"text": "your natural language question"},
    "retriever": {"query": "keyword terms"},
})
```

### Metadata filter

Retrieve documents by metadata filters only.

```python
from haystack_integrations.components.topk import TopKMetadataRetriever

retriever = TopKMetadataRetriever(document_store=store, top_k=5)
pipeline.add_component("retriever", retriever)
result = pipeline.run({"retriever": {"filters": {
    "operator": "AND",
    "conditions": [
        {"field": "meta.language", "operator": "==", "value": "en"},
        {"field": "meta.year", "operator": ">=", "value": 2020},
    ],
}}})
```

## Metadata filters

All retrievers accept Haystack-style filter dicts. Supported operators:

| Operator | Description |
|---|---|
| `==`, `!=` | Equality / inequality |
| `>`, `>=`, `<`, `<=` | Numeric comparison |
| `in` | Field value is in a list |
| `not in` | Field value is not in a list |
| `AND`, `OR`, `NOT` | Logical combinators |

```python
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.language", "operator": "==", "value": "en"},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.year", "operator": "==", "value": 2025},
            ],
        },
    ],
}
```

## Multi-tenant (partitions)

Use the `partition` parameter to scope all reads and writes to a logical partition. Different partitions in the same collection are fully isolated.

```python
store_a = TopKDocumentStore(region="...", collection_name="shared", partition="tenant-a")
store_b = TopKDocumentStore(region="...", collection_name="shared", partition="tenant-b")
```

## Development

```bash
git clone https://github.com/topk-io/topk-haystack
cd topk-haystack
uv sync --group dev
```

```bash
export TOPK_API_KEY=your-api-key   # https://console.topk.io/api-key
export TOPK_REGION=aws-us-east-1-elastica
```

```bash
uv run pytest -m "not integration" tests/   # unit tests
uv run pytest -m "integration" tests/       # integration tests
uv run pytest tests/                        # all tests
uv run pytest --cov=haystack_integrations tests/  # with coverage
```

Lint and format:

```bash
uv run ruff check --fix . && uv run ruff format .   # auto-fix
uv run ruff check . && uv run ruff format --check . # check only
```

Or with [Hatch](https://hatch.pypa.io/):

```bash
hatch run fmt            # auto-fix
hatch run fmt-check      # check only
hatch run test:unit
hatch run test:integration
hatch run test:all
hatch run test:cov
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
