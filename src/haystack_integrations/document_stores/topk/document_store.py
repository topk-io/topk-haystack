# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream, Document, SparseEmbedding
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from topk_sdk import Client, CollectionClient
from topk_sdk.data import f32_sparse_vector as f32_sparse_vector_data
from topk_sdk.error import CollectionAlreadyExistsError, CollectionNotFoundError
from topk_sdk.query import select
from topk_sdk.schema import bytes as topk_bytes
from topk_sdk.schema import f32_sparse_vector, f32_vector, keyword_index, semantic_index, text, vector_index

from haystack_integrations.document_stores.topk.filters import (
    extract_meta_fields,
    reconstruct_meta,
    translate_filters,
)

logger = logging.getLogger(__name__)


def _document_to_topk(doc: Document) -> dict[str, Any]:
    """Convert a Haystack Document to a TopK document dict."""
    data: dict[str, Any] = {"_id": doc.id}

    if doc.content is not None:
        data["content"] = doc.content

    if doc.embedding is not None:
        data["embedding"] = doc.embedding

    if doc.sparse_embedding is not None:
        data["sparse_embedding"] = f32_sparse_vector_data(
            dict(zip(doc.sparse_embedding.indices, doc.sparse_embedding.values, strict=True))
        )

    if doc.meta is not None:
        data["meta"] = doc.meta

    if doc.blob is not None:
        data["blob"] = doc.blob.data

    return data


def _topk_to_document(result: dict[str, Any], meta_fields: list[str] | None = None) -> Document:
    """Convert a TopK query result dict to a Haystack Document."""
    doc_id = result.get("_id", "")
    content = result.get("content")
    score = result.get("score")
    embedding = result.get("embedding")
    blob_data = result.get("blob")

    blob = None
    if blob_data is not None:
        blob = ByteStream(data=blob_data)

    sparse_raw = result.get("sparse_embedding")
    sparse_embedding = None
    if sparse_raw is not None:
        sparse_embedding = SparseEmbedding(indices=sparse_raw["indices"], values=sparse_raw["values"])

    meta = reconstruct_meta(result, meta_fields) if meta_fields else {}

    return Document(
        id=doc_id,
        content=content,
        meta=meta,
        score=score,
        embedding=embedding,
        blob=blob,
        sparse_embedding=sparse_embedding,
    )


class TopKDocumentStore:
    """
    Haystack DocumentStore backed by a TopK collection.

    Supports BM25 keyword search, dense vector search, server-side semantic search,
    and hybrid (vector + BM25) search via the companion Retriever components.

    TopK uses upsert semantics — writes always overwrite existing documents with the same ID.
    ``DuplicatePolicy.SKIP`` and ``FAIL`` are not supported and log a warning if passed.

    Meta fields are stored and filterable but only returned in query results when explicitly
    referenced in the filter — TopK requires field paths to be whitelisted in the select stage.
    """

    def __init__(
        self,
        region: str,
        api_key: Secret = Secret.from_env_var("TOPK_API_KEY"),  # noqa: B008
        collection_name: str = "haystack",
        embedding_dim: int = 768,
        similarity: Literal["cosine", "euclidean", "dot_product"] = "cosine",
        recreate_collection: bool = False,
        host: str = "topk.io",
        https: bool = True,
        filter_documents_limit: int = 10_000,
        partition: str | None = None,
    ) -> None:
        """
        Initialize the TopK document store.

        :param region: TopK region identifier. See available regions: https://topk.io/docs/regions
        :param api_key: TopK API key. Defaults to ``TOPK_API_KEY`` env var. Get your API key from the TopK console: https://console.topk.io/api-key
        :param collection_name: Name of the TopK collection to use.
        :param embedding_dim: Dimensionality of dense embedding vectors.
        :param similarity: Vector similarity metric — ``"cosine"``, ``"euclidean"``, or ``"dot_product"``.
        :param recreate_collection: Drop and recreate the collection on init if it already exists.
        :param host: TopK API host. Defaults to ``"topk.io"``.
        :param https: Whether to use HTTPS. Defaults to ``True``.
        :param filter_documents_limit: Maximum number of documents returned by ``filter_documents``.
            Defaults to 10 000.
        :param partition: Optional partition name for multi-tenant use. All reads and writes are
            scoped to this partition. ``None`` uses the default (unpartitioned) partition.
        """
        if embedding_dim <= 0:
            msg = f"embedding_dim must be a positive integer, got {embedding_dim}"
            raise ValueError(msg)

        self.api_key = api_key
        self.region = region
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.similarity = similarity
        self.recreate_collection = recreate_collection
        self.host = host
        self.https = https
        self.filter_documents_limit = filter_documents_limit
        self.partition = partition

        self._client = Client(api_key=api_key.resolve_value(), region=region, host=host, https=https)
        self._ensure_collection(recreate=recreate_collection)

    def _ensure_collection(self, recreate: bool = False) -> None:
        if recreate:
            try:
                self._client.collections().delete(self.collection_name)
            except CollectionNotFoundError:
                pass

        schema = {
            "content": text().index(keyword_index()).index(semantic_index()),
            "embedding": f32_vector(self.embedding_dim).index(vector_index(metric=self.similarity)),
            "sparse_embedding": f32_sparse_vector(),
            "blob": topk_bytes().optional(),
        }

        try:
            self._client.collections().create(self.collection_name, schema)
        except CollectionAlreadyExistsError:
            pass

    def _collection(self) -> CollectionClient:
        return self._client.collection(self.collection_name, partition=self.partition)

    def count_documents(self) -> int:
        """Return the number of documents in the store."""
        return self._collection().count()

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Return documents matching the given Haystack filter dict.

        Meta fields referenced in ``filters`` are automatically included in the result.

        :param filters: Haystack filter dict. If ``None`` or ``{}``, all documents are returned.
        :returns: List of matching Documents with any filtered meta fields populated.
        """
        expr = translate_filters(filters) if filters else None
        meta_fields = extract_meta_fields(filters)

        query = select("content", "blob", *meta_fields)
        if expr is not None:
            query = query.filter(expr)
        query = query.limit(self.filter_documents_limit)

        results = self._collection().query(query)
        return [_topk_to_document(r, meta_fields=meta_fields) for r in results]

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Write documents to the store.

        TopK only supports upsert — all writes overwrite existing documents with the same ID.
        Passing ``SKIP`` or ``FAIL`` logs a warning and falls back to upsert.

        :param documents: Documents to write.
        :param policy: Accepted for interface compatibility; only upsert semantics are applied.
        :returns: Number of documents written.
        """
        if not documents:
            return 0

        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            logger.warning(
                "TopKDocumentStore only supports upsert. DuplicatePolicy.%s is not supported and will be ignored.",
                policy.name,
            )

        self._collection().upsert([_document_to_topk(d) for d in documents])
        return len(documents)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return
        self._collection().delete(document_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document store to a dict."""
        return default_to_dict(
            self,
            region=self.region,
            api_key=self.api_key.to_dict(),
            collection_name=self.collection_name,
            embedding_dim=self.embedding_dim,
            similarity=self.similarity,
            recreate_collection=False,
            host=self.host,
            https=self.https,
            filter_documents_limit=self.filter_documents_limit,
            partition=self.partition,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopKDocumentStore":
        """Deserialize a document store from a dict."""
        if api_key := data.get("init_parameters", {}).get("api_key"):
            data["init_parameters"]["api_key"] = Secret.from_dict(api_key)
        return default_from_dict(cls, data)
