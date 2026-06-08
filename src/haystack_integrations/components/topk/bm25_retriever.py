# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy, apply_filter_policy
from topk_sdk.query import field, fn, match, select

from haystack_integrations.document_stores.topk.document_store import TopKDocumentStore, _topk_to_document
from haystack_integrations.document_stores.topk.filters import extract_meta_fields, translate_filters


@component
class TopKBM25Retriever:
    """
    Retriever that performs BM25 keyword search against a TopKDocumentStore.
    """

    def __init__(
        self,
        document_store: TopKDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the retriever.

        :param document_store: A ``TopKDocumentStore`` instance.
        :param filters: Default metadata filters applied to every query.
        :param top_k: Number of documents to return.
        :param filter_policy: How to merge init-time and run-time filters.
        """
        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k
        self._filter_policy = filter_policy

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run a BM25 keyword search.

        :param query: Text query string.
        :param filters: Run-time metadata filters; merged with init filters per ``filter_policy``.
        :param top_k: Override the number of results to return.
        :returns: ``{"documents": [...]}``
        """
        filters_merged = apply_filter_policy(self._filter_policy, self._filters, filters)
        effective_k = top_k if top_k is not None else self._top_k
        meta_fields = extract_meta_fields(filters_merged)

        query_builder = select("content", "blob", "blob_mime_type", *meta_fields, score=fn.bm25_score())
        query_builder = query_builder.filter(match(query, field="content"))

        if filters_merged:
            expr = translate_filters(filters_merged)
            if expr is not None:
                query_builder = query_builder.filter(expr)

        query_builder = query_builder.topk(field("score"), effective_k, asc=False)

        results = self._document_store._collection().query(query_builder)
        return {"documents": [_topk_to_document(r, meta_fields=meta_fields) for r in results]}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retriever to a dict."""
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopKBM25Retriever":
        """Deserialize a retriever from a dict."""
        data["init_parameters"]["document_store"] = TopKDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        fp = data["init_parameters"].get("filter_policy")
        if fp is not None:
            data["init_parameters"]["filter_policy"] = FilterPolicy(fp)
        return default_from_dict(cls, data)
