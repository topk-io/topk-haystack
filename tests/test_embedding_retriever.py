# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.topk import TopKEmbeddingRetriever
from haystack_integrations.document_stores.topk import TopKDocumentStore


class TestTopKEmbeddingRetrieverUnit:
    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock(spec=TopKDocumentStore)
        store.collection_name = "test_collection"
        store._collection.return_value.query.return_value = [
            {"_id": "1", "content": "hello", "score": 0.9, "meta.year": 2024}
        ]
        return store

    def test_run_returns_documents(self, mock_store: MagicMock) -> None:
        retriever = TopKEmbeddingRetriever(document_store=mock_store, top_k=5)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "hello"

    def test_run_respects_top_k_override(self, mock_store: MagicMock) -> None:
        retriever = TopKEmbeddingRetriever(document_store=mock_store, top_k=5)
        retriever.run(query_embedding=[0.1, 0.2], top_k=3)
        mock_store._collection.assert_called()

    def test_run_with_filters(self, mock_store: MagicMock) -> None:
        retriever = TopKEmbeddingRetriever(document_store=mock_store)
        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"field": "meta.year", "operator": "==", "value": 2024},
        )
        assert "documents" in result
        assert result["documents"][0].meta == {"year": 2024}

    def test_run_selects_filtered_meta_fields(self, mock_store: MagicMock) -> None:
        retriever = TopKEmbeddingRetriever(document_store=mock_store)
        query = MagicMock()
        query.filter.return_value = query
        query.topk.return_value = query
        with patch(
            "haystack_integrations.components.topk.embedding_retriever.select", return_value=query
        ) as mock_select:
            retriever.run(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                filters={"field": "meta.year", "operator": "==", "value": 2024},
            )
        assert mock_select.call_args.args == ("content", "blob", "blob_mime_type", "meta.year")

    def test_to_dict(self, mock_store: MagicMock) -> None:
        retriever = TopKEmbeddingRetriever(
            document_store=mock_store,
            filters={"field": "meta.year", "operator": "==", "value": 2024},
            top_k=7,
        )
        d = retriever.to_dict()
        assert d["type"].endswith("TopKEmbeddingRetriever")
        params = d["init_parameters"]
        assert params["top_k"] == 7
        assert params["filter_policy"] == "replace"
        assert params["filters"] == {"field": "meta.year", "operator": "==", "value": 2024}
        assert "document_store" in params

    def test_from_dict_round_trip(self, mock_store: MagicMock) -> None:
        mock_store.to_dict.return_value = {
            "type": "haystack_integrations.document_stores.topk.document_store.TopKDocumentStore",
            "init_parameters": {
                "region": "aws-us-east-1-elastica",
                "api_key": {"type": "env_var", "env_vars": ["TOPK_API_KEY"], "strict": True},
                "collection_name": "haystack",
                "embedding_dim": 768,
                "similarity": "cosine",
                "recreate_collection": False,
                "host": "topk.io",
                "https": True,
                "filter_documents_limit": 10000,
            },
        }
        retriever = TopKEmbeddingRetriever(
            document_store=mock_store,
            filters={"field": "meta.year", "operator": "==", "value": 2024},
            top_k=7,
            filter_policy=FilterPolicy.MERGE,
        )
        with patch.object(TopKDocumentStore, "from_dict", return_value=mock_store):
            restored = TopKEmbeddingRetriever.from_dict(retriever.to_dict())
        assert restored._top_k == 7
        assert restored._filters == {"field": "meta.year", "operator": "==", "value": 2024}
        assert restored._filter_policy == FilterPolicy.MERGE


@pytest.mark.integration
class TestTopKEmbeddingRetrieverIntegration:
    def test_retrieval(self, topk_store: TopKDocumentStore) -> None:
        docs = [
            Document(content="Python programming", meta={"lang": "python"}, embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Rust systems", meta={"lang": "rust"}, embedding=[0.9, 0.8, 0.7, 0.6]),
        ]
        topk_store.write_documents(docs)

        retriever = TopKEmbeddingRetriever(document_store=topk_store, top_k=2)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert len(result["documents"]) > 0
        assert all(isinstance(d, Document) for d in result["documents"])
