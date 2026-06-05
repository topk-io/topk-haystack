# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.topk import TopKSemanticRetriever
from haystack_integrations.document_stores.topk import TopKDocumentStore


class TestTopKSemanticRetrieverUnit:
    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock(spec=TopKDocumentStore)
        store.collection_name = "test_collection"
        store._collection.return_value.query.return_value = [{"_id": "1", "content": "Python is great", "score": 0.95}]
        return store

    def test_run_returns_documents(self, mock_store: MagicMock) -> None:
        retriever = TopKSemanticRetriever(document_store=mock_store, top_k=5)
        result = retriever.run(query="what is Python?")
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0].score == pytest.approx(0.95)

    def test_run_with_filters(self, mock_store: MagicMock) -> None:
        retriever = TopKSemanticRetriever(document_store=mock_store)
        result = retriever.run(
            query="memory safe",
            filters={"field": "meta.lang", "operator": "==", "value": "rust"},
        )
        assert "documents" in result

    def test_to_dict(self, mock_store: MagicMock) -> None:
        retriever = TopKSemanticRetriever(
            document_store=mock_store,
            filters={"field": "meta.lang", "operator": "==", "value": "rust"},
            top_k=7,
        )
        d = retriever.to_dict()
        assert d["type"].endswith("TopKSemanticRetriever")
        params = d["init_parameters"]
        assert params["top_k"] == 7
        assert params["filter_policy"] == "replace"
        assert params["filters"] == {"field": "meta.lang", "operator": "==", "value": "rust"}
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
        retriever = TopKSemanticRetriever(
            document_store=mock_store,
            filters={"field": "meta.lang", "operator": "==", "value": "rust"},
            top_k=7,
            filter_policy=FilterPolicy.MERGE,
        )
        with patch.object(TopKDocumentStore, "from_dict", return_value=mock_store):
            restored = TopKSemanticRetriever.from_dict(retriever.to_dict())
        assert restored._top_k == 7
        assert restored._filters == {"field": "meta.lang", "operator": "==", "value": "rust"}
        assert restored._filter_policy == FilterPolicy.MERGE


@pytest.mark.integration
class TestTopKSemanticRetrieverIntegration:
    def test_semantic_retrieval_no_embedder(self, topk_store: TopKDocumentStore) -> None:
        docs = [
            Document(content="Python is a great programming language"),
            Document(content="Rust is fast and memory safe"),
        ]
        topk_store.write_documents(docs)

        retriever = TopKSemanticRetriever(document_store=topk_store, top_k=2)
        result = retriever.run(query="memory safe language")
        assert len(result["documents"]) > 0
        assert all(isinstance(d, Document) for d in result["documents"])
