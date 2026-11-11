"""OpenTelemetry helpers for MCP."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any, Final

from mcp_types import INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import SpanKind, get_current_span, get_tracer
from opentelemetry.trace.span import Span

_tracer = get_tracer("mcp-python-sdk")

# JSON-RPC codes that are caller/policy errors, not operation failures: the request
# was malformed, invalid, or unsatisfiable. The OpenTelemetry semantic conventions
# treat these as non-error outcomes, so their spans stay UNSET with no `error.type`.
# -32002 is the retired resource-not-found code (reserved-never-reused, so no
# `mcp_types` constant) that handlers still emit for missing resources.
_CALLER_ERROR_CODES: Final[frozenset[int]] = frozenset(
    {PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, -32002}
)


def is_caller_error(code: int) -> bool:
    """Whether a JSON-RPC error `code` is a caller/policy error rather than an operation failure.

    Spans for caller errors record `rpc.response.status_code` only: the OTel semantic
    conventions leave the span status unset and omit `error.type`, which must carry a
    semconv registry value rather than the wire code.
    """
    return code in _CALLER_ERROR_CODES


@contextmanager
def otel_span(
    name: str,
    *,
    kind: SpanKind,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Generator[Span]:
    """Create an OTel span."""
    with _tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
        context=context,
        record_exception=record_exception,
        set_status_on_exception=set_status_on_exception,
    ) as span:
        yield span


def inject_trace_context(meta: dict[str, Any]) -> None:
    """Inject W3C trace context (traceparent/tracestate) into a `_meta` dict."""
    inject(meta)


def extract_trace_context(meta: Mapping[str, Any] | None) -> Context | None:
    """Extract W3C trace context from a `_meta` dict.

    Returns `None` when the carrier is absent, malformed, or carries no
    valid `traceparent`, so callers fall through to ambient parenting; an
    explicit empty `Context` would orphan the span instead of nesting under
    the current one.
    """
    if not meta:
        return None
    try:
        ctx = extract(meta)
    except (ValueError, TypeError):
        return None
    if not get_current_span(ctx).get_span_context().is_valid:
        return None
    return ctx
