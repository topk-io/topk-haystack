# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.topk import TopKDocumentStore


class TestTopKDocumentStoreUnit:
    @pytest.fixture
    def mock_store(self) -> TopKDocumentStore:
        with (
            patch("haystack_integrations.document_stores.topk.document_store.Client"),
            patch.object(TopKDocumentStore, "_ensure_collection"),
        ):
            store = TopKDocumentStore(region="aws-us-east-1-elastica", api_key=Secret.from_token("fake"))
        store._client = MagicMock()
        return store

    def test_filter_documents_no_filter_returns_documents(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = [
            {"_id": "1", "content": "hello"},
            {"_id": "2", "content": "world"},
        ]
        result = mock_store.filter_documents()
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_filter_documents_selects_content_blob_and_blob_mime_type(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents()
        mock_select.assert_called_once_with("content", "blob", "blob_mime_type")

    def test_filter_documents_applies_default_limit(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents()
        mock_query.limit.assert_called_once_with(10_000)

    def test_filter_documents_respects_custom_limit(self, mock_store: TopKDocumentStore) -> None:
        mock_store.filter_documents_limit = 500
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents()
        mock_query.limit.assert_called_once_with(500)

    def test_filter_documents_with_filter_calls_filter_on_query(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": "python"})
        mock_query.filter.assert_called_once()

    def test_filter_documents_meta_field_added_to_select(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": "python"})
        mock_select.assert_called_once_with("content", "blob", "blob_mime_type", "meta.lang")

    def test_filter_documents_reconstructs_meta(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = [
            {"_id": "1", "content": "doc", "meta.lang": "python"}
        ]
        result = mock_store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": "python"})
        assert result[0].meta == {"lang": "python"}

    def test_filter_documents_reconstructs_blob_mime_type(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = [
            {"_id": "1", "blob": b"%PDF", "blob_mime_type": "application/pdf"}
        ]
        result = mock_store.filter_documents()
        assert result[0].blob is not None
        assert result[0].blob.data == b"%PDF"
        assert result[0].blob.mime_type == "application/pdf"

    def test_write_documents_stores_blob_mime_type(self, mock_store: TopKDocumentStore) -> None:
        doc = Document(blob=ByteStream(data=b"%PDF", mime_type="application/pdf"))
        mock_store.write_documents([doc])
        written = mock_store._client.collection.return_value.upsert.call_args.args[0][0]
        assert written["_id"] == doc.id
        assert written["blob"] == b"%PDF"
        assert written["blob_mime_type"] == "application/pdf"

    def test_invalid_embedding_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            with (
                patch("haystack_integrations.document_stores.topk.document_store.Client"),
                patch.object(TopKDocumentStore, "_ensure_collection"),
            ):
                TopKDocumentStore(region="aws-us-east-1-elastica", api_key=Secret.from_token("fake"), embedding_dim=0)

    def test_negative_embedding_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            with (
                patch("haystack_integrations.document_stores.topk.document_store.Client"),
                patch.object(TopKDocumentStore, "_ensure_collection"),
            ):
                TopKDocumentStore(region="aws-us-east-1-elastica", api_key=Secret.from_token("fake"), embedding_dim=-1)

    def test_schema_includes_blob_mime_type(self) -> None:
        with patch("haystack_integrations.document_stores.topk.document_store.Client") as mock_client:
            TopKDocumentStore(region="aws-us-east-1-elastica", api_key=Secret.from_token("fake"))
        schema = mock_client.return_value.collections.return_value.create.call_args.args[1]
        assert "blob_mime_type" in schema

    def test_filter_documents_queries_correct_collection(self, mock_store: TopKDocumentStore) -> None:
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents()
        mock_store._client.collection.assert_called_with(mock_store.collection_name, partition=None)

    def test_partition_passed_to_collection(self, mock_store: TopKDocumentStore) -> None:
        mock_store.partition = "tenant_a"
        mock_store._client.collection.return_value.query.return_value = []
        with patch("haystack_integrations.document_stores.topk.document_store.select") as mock_select:
            mock_query = MagicMock()
            mock_select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_store.filter_documents()
        mock_store._client.collection.assert_called_with(mock_store.collection_name, partition="tenant_a")

    def test_to_dict_includes_partition(self) -> None:
        with (
            patch.dict("os.environ", {"TOPK_API_KEY": "fake"}),
            patch("haystack_integrations.document_stores.topk.document_store.Client"),
            patch.object(TopKDocumentStore, "_ensure_collection"),
        ):
            store = TopKDocumentStore(region="aws-us-east-1-elastica", partition="tenant_a")
        d = store.to_dict()
        assert d["init_parameters"]["partition"] == "tenant_a"

    def test_to_dict_partition_none_by_default(self) -> None:
        with (
            patch.dict("os.environ", {"TOPK_API_KEY": "fake"}),
            patch("haystack_integrations.document_stores.topk.document_store.Client"),
            patch.object(TopKDocumentStore, "_ensure_collection"),
        ):
            store = TopKDocumentStore(region="aws-us-east-1-elastica")
        d = store.to_dict()
        assert d["init_parameters"]["partition"] is None

    def test_duplicate_policy_fail_raises(self, mock_store: TopKDocumentStore) -> None:
        with pytest.raises(ValueError, match=r"DuplicatePolicy\.FAIL"):
            mock_store.write_documents([Document(content="doc")], policy=DuplicatePolicy.FAIL)
        mock_store._client.collection.assert_not_called()

    def test_duplicate_policy_skip_raises(self, mock_store: TopKDocumentStore) -> None:
        with pytest.raises(ValueError, match=r"DuplicatePolicy\.SKIP"):
            mock_store.write_documents([Document(content="doc")], policy=DuplicatePolicy.SKIP)
        mock_store._client.collection.assert_not_called()


@pytest.mark.integration
class TestTopKDocumentStore:
    @pytest.fixture
    def document_store(self, topk_store: TopKDocumentStore) -> TopKDocumentStore:
        return topk_store

    def test_count_empty(self, document_store: TopKDocumentStore) -> None:
        assert document_store.count_documents() == 0

    def test_write_and_count(self, document_store: TopKDocumentStore) -> None:
        docs = [Document(content="doc 1"), Document(content="doc 2")]
        written = document_store.write_documents(docs)
        assert written == 2
        assert document_store.count_documents() == 2

    def test_write_empty_returns_zero(self, document_store: TopKDocumentStore) -> None:
        assert document_store.write_documents([]) == 0

    def test_filter_all(self, document_store: TopKDocumentStore) -> None:
        docs = [Document(content="a"), Document(content="b")]
        document_store.write_documents(docs)
        result = document_store.filter_documents()
        assert len(result) == 2

    def test_filter_by_metadata(self, document_store: TopKDocumentStore) -> None:
        docs = [
            Document(content="python", meta={"lang": "python"}),
            Document(content="rust", meta={"lang": "rust"}),
        ]
        document_store.write_documents(docs)
        result = document_store.filter_documents(filters={"field": "meta.lang", "operator": "==", "value": "python"})
        assert len(result) == 1
        assert result[0].content == "python"
        assert result[0].meta == {"lang": "python"}

    def test_filter_empty_dict_returns_all(self, document_store: TopKDocumentStore) -> None:
        document_store.write_documents([Document(content="a"), Document(content="b")])
        result = document_store.filter_documents(filters={})
        assert len(result) == 2

    def test_delete_documents(self, document_store: TopKDocumentStore) -> None:
        doc = Document(content="to delete")
        document_store.write_documents([doc])
        document_store.delete_documents([doc.id])
        assert document_store.count_documents() == 0

    def test_delete_empty_list(self, document_store: TopKDocumentStore) -> None:
        document_store.delete_documents([])  # should not raise

    def test_duplicate_policy_overwrite(self, document_store: TopKDocumentStore) -> None:
        doc = Document(id="dup-id", content="original")
        document_store.write_documents([doc])
        doc2 = Document(id="dup-id", content="updated")
        document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE)
        assert document_store.count_documents() == 1

    def test_duplicate_policy_fail_raises(self, document_store: TopKDocumentStore) -> None:
        doc = Document(id="fail-id", content="first")
        document_store.write_documents([doc])
        with pytest.raises(ValueError, match=r"DuplicatePolicy\.FAIL"):
            document_store.write_documents([Document(id="fail-id", content="second")], policy=DuplicatePolicy.FAIL)
        assert document_store.count_documents() == 1

    def test_duplicate_policy_skip_raises(self, document_store: TopKDocumentStore) -> None:
        doc = Document(id="skip-id", content="first")
        document_store.write_documents([doc])
        with pytest.raises(ValueError, match=r"DuplicatePolicy\.SKIP"):
            document_store.write_documents([Document(id="skip-id", content="second")], policy=DuplicatePolicy.SKIP)
        assert document_store.count_documents() == 1

    def test_to_dict_from_dict_roundtrip(self, document_store: TopKDocumentStore) -> None:
        data = document_store.to_dict()
        assert data["type"].endswith("TopKDocumentStore")
        restored = TopKDocumentStore.from_dict(data)
        assert restored.collection_name == document_store.collection_name
        assert restored.embedding_dim == document_store.embedding_dim
        assert restored.similarity == document_store.similarity

    def test_filter_with_none_returns_all(self, document_store: TopKDocumentStore) -> None:
        document_store.write_documents([Document(content="x"), Document(content="y")])
        result = document_store.filter_documents(filters=None)
        assert len(result) == 2

    def test_documents_with_embedding(self, document_store: TopKDocumentStore) -> None:
        doc = Document(content="vec doc", embedding=[0.1, 0.2, 0.3, 0.4])
        document_store.write_documents([doc])
        result = document_store.filter_documents()
        assert len(result) == 1

    def test_documents_with_metadata_fields(self, document_store: TopKDocumentStore) -> None:
        doc = Document(content="meta doc", meta={"year": 2024, "tags": ["a", "b"]})
        document_store.write_documents([doc])
        result = document_store.filter_documents()
        # Meta is stored and filterable but not returned in query results (schemaless fields
        # require explicit whitelisting in select — see TopKDocumentStore class docstring).
        assert len(result) == 1
        assert result[0].content == "meta doc"

    def test_blob_mime_type_roundtrip(self, document_store: TopKDocumentStore) -> None:
        doc = Document(blob=ByteStream(data=b"%PDF-1.4", mime_type="application/pdf"))
        document_store.write_documents([doc])
        result = document_store.filter_documents()
        assert len(result) == 1
        assert result[0].blob is not None
        assert result[0].blob.data == b"%PDF-1.4"
        assert result[0].blob.mime_type == "application/pdf"
