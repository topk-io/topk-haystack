# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from uuid import uuid4

import pytest

from haystack_integrations.document_stores.topk import TopKDocumentStore


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires TOPK_API_KEY env var")


@pytest.fixture
def topk_store() -> Generator[TopKDocumentStore, None, None]:
    """Provide a TopKDocumentStore backed by a unique test collection, cleaned up after the test."""
    api_key = os.environ.get("TOPK_API_KEY")
    if not api_key:
        pytest.skip("TOPK_API_KEY not set")

    collection_name = f"test_{uuid4().hex[:8]}"
    store = TopKDocumentStore(
        region=os.environ.get("TOPK_REGION", "aws-us-east-1-elastica"),
        collection_name=collection_name,
        embedding_dim=4,
    )
    yield store

    try:
        store._client.collections().delete(collection_name)
    except Exception:
        pass
