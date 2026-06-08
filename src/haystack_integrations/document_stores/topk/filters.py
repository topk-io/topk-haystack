# SPDX-FileCopyrightText: 2026-present TopK Team <support@topk.io>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.errors import FilterError
from topk_sdk.query import LogicalExpr, field, not_
from topk_sdk.query import all as topk_all
from topk_sdk.query import any as topk_any


def extract_meta_fields(filters: dict[str, Any] | None) -> list[str]:
    """
    Return deduplicated list of meta sub-field paths referenced in a filter dict.

    E.g. ``{"field": "meta.year", ...}`` → ``["meta.year"]``
    """
    if not filters:
        return []
    return list(dict.fromkeys(_collect_meta_fields(filters)))


def _collect_meta_fields(filters: dict[str, Any]) -> list[str]:
    """Recursively walk a filter tree and collect all ``meta.*`` field paths, including duplicates."""
    if "conditions" in filters:
        return [f for c in filters["conditions"] for f in _collect_meta_fields(c)]
    if "field" in filters:
        field_name: str = filters["field"]
        return [field_name] if field_name.startswith("meta.") else []
    return []


def reconstruct_meta(result: dict[str, Any], meta_fields: list[str]) -> dict[str, Any]:
    """
    Build a nested meta dict from the flat dotted keys TopK returns.

    ``"meta.address.city"`` → ``{"address": {"city": ...}}``
    """
    meta: dict[str, Any] = {}
    for path in meta_fields:
        if path in result:
            keys = path[len("meta.") :].split(".")
            target = meta
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = result[path]
    return meta


def translate_filters(filters: dict[str, Any]) -> LogicalExpr | None:
    """Translate a Haystack filter dict into a TopK query expression."""
    if not filters:
        return None

    if "operator" not in filters:
        msg = f"Invalid filter: missing 'operator' key in {filters}"
        raise FilterError(msg)

    operator = filters["operator"]

    if operator in ("AND", "OR", "NOT"):
        return _translate_logical(filters)
    return _translate_comparison(filters)


def _translate_logical(filters: dict[str, Any]) -> LogicalExpr:
    operator = filters["operator"]
    conditions = filters.get("conditions", [])

    if not conditions:
        msg = f"Logical filter '{operator}' has no conditions"
        raise FilterError(msg)

    translated = [translate_filters(c) for c in conditions]

    if operator == "AND":
        return topk_all(translated)
    if operator == "OR":
        return topk_any(translated)
    if operator == "NOT":
        if len(translated) != 1:
            msg = "NOT filter must have exactly one condition"
            raise FilterError(msg)
        return not_(translated[0])
    msg = f"Unknown logical operator: {operator}"
    raise FilterError(msg)


def _translate_comparison(filters: dict[str, Any]) -> LogicalExpr:
    if "field" not in filters or "operator" not in filters:
        msg = f"Invalid comparison filter: {filters}"
        raise FilterError(msg)

    if "value" not in filters:
        msg = f"Comparison filter is missing 'value' key: {filters}"
        raise FilterError(msg)

    field_name: str = filters["field"]
    operator: str = filters["operator"]
    value = filters["value"]

    f = field(field_name)

    if operator == "==":
        return f.eq(value)
    if operator == "!=":
        return f.ne(value)
    if operator == ">":
        return f.gt(value)
    if operator == ">=":
        return f.gte(value)
    if operator == "<":
        return f.lt(value)
    if operator == "<=":
        return f.lte(value)
    if operator == "in":
        if not isinstance(value, list):
            msg = f"'in' operator requires a list value, got {type(value)}"
            raise FilterError(msg)
        if not value:
            msg = "'in' operator requires a non-empty list"
            raise FilterError(msg)
        return f.in_(value)
    if operator == "not in":
        if not isinstance(value, list):
            msg = f"'not in' operator requires a list value, got {type(value)}"
            raise FilterError(msg)
        if not value:
            msg = "'not in' operator requires a non-empty list"
            raise FilterError(msg)
        return topk_all([field(field_name).ne(v) for v in value])

    msg = f"Unknown comparison operator: {operator}"
    raise FilterError(msg)
