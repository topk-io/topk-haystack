# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.topk.filters import reconstruct_meta, translate_filters


class TestTranslateFilters:
    def test_eq(self) -> None:
        result = translate_filters({"field": "meta.year", "operator": "==", "value": 2024})
        assert result is not None

    def test_ne(self) -> None:
        result = translate_filters({"field": "meta.genre", "operator": "!=", "value": "fiction"})
        assert result is not None

    def test_gt(self) -> None:
        result = translate_filters({"field": "meta.score", "operator": ">", "value": 0.5})
        assert result is not None

    def test_gte(self) -> None:
        result = translate_filters({"field": "meta.score", "operator": ">=", "value": 0.5})
        assert result is not None

    def test_lt(self) -> None:
        result = translate_filters({"field": "meta.score", "operator": "<", "value": 1.0})
        assert result is not None

    def test_lte(self) -> None:
        result = translate_filters({"field": "meta.score", "operator": "<=", "value": 1.0})
        assert result is not None

    def test_in(self) -> None:
        result = translate_filters({"field": "meta.genre", "operator": "in", "value": ["fiction", "history"]})
        assert result is not None

    def test_not_in(self) -> None:
        result = translate_filters({"field": "meta.genre", "operator": "not in", "value": ["fiction", "history"]})
        assert result is not None

    def test_and(self) -> None:
        result = translate_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.year", "operator": ">=", "value": 2020},
                    {"field": "meta.genre", "operator": "==", "value": "fiction"},
                ],
            }
        )
        assert result is not None

    def test_or(self) -> None:
        result = translate_filters(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.year", "operator": "==", "value": 2020},
                    {"field": "meta.year", "operator": "==", "value": 2021},
                ],
            }
        )
        assert result is not None

    def test_not(self) -> None:
        result = translate_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.genre", "operator": "==", "value": "fiction"},
                ],
            }
        )
        assert result is not None

    def test_none_returns_none(self) -> None:
        assert translate_filters(None) is None  # type: ignore[arg-type]

    def test_missing_operator_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "value": 2020})

    def test_in_with_non_list_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "in", "value": 2020})

    def test_not_in_with_non_list_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "not in", "value": 2020})

    def test_not_in_with_empty_list_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "not in", "value": []})

    def test_in_with_empty_list_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "in", "value": []})

    def test_unknown_operator_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "~=", "value": 2020})

    def test_missing_value_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"field": "meta.year", "operator": "=="})

    def test_empty_and_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters({"operator": "AND", "conditions": []})

    def test_not_multiple_conditions_raises(self) -> None:
        with pytest.raises(FilterError):
            translate_filters(
                {
                    "operator": "NOT",
                    "conditions": [
                        {"field": "meta.a", "operator": "==", "value": 1},
                        {"field": "meta.b", "operator": "==", "value": 2},
                    ],
                }
            )

    def test_not_and(self) -> None:
        result = translate_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.year", "operator": ">=", "value": 2020},
                            {"field": "meta.genre", "operator": "==", "value": "fiction"},
                        ],
                    }
                ],
            }
        )
        assert result is not None

    def test_not_or(self) -> None:
        result = translate_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.genre", "operator": "==", "value": "fiction"},
                            {"field": "meta.genre", "operator": "==", "value": "history"},
                        ],
                    }
                ],
            }
        )
        assert result is not None

    def test_nested_and_or(self) -> None:
        result = translate_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.year", "operator": ">=", "value": 2020},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.genre", "operator": "==", "value": "fiction"},
                            {"field": "meta.genre", "operator": "==", "value": "history"},
                        ],
                    },
                ],
            }
        )
        assert result is not None


class TestReconstructMeta:
    def test_single_level(self) -> None:
        result = reconstruct_meta({"meta.lang": "python"}, ["meta.lang"])
        assert result == {"lang": "python"}

    def test_nested_two_levels(self) -> None:
        result = reconstruct_meta({"meta.address.city": "Berlin"}, ["meta.address.city"])
        assert result == {"address": {"city": "Berlin"}}

    def test_nested_three_levels(self) -> None:
        result = reconstruct_meta({"meta.a.b.c": 42}, ["meta.a.b.c"])
        assert result == {"a": {"b": {"c": 42}}}

    def test_multiple_fields_share_parent(self) -> None:
        result = reconstruct_meta(
            {"meta.address.city": "Berlin", "meta.address.zip": "10115"},
            ["meta.address.city", "meta.address.zip"],
        )
        assert result == {"address": {"city": "Berlin", "zip": "10115"}}

    def test_missing_key_skipped(self) -> None:
        result = reconstruct_meta({}, ["meta.lang"])
        assert result == {}
