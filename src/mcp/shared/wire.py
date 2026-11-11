"""Outbound JSON-RPC wire framing, with per-message failure containment (#3707).

A `str` can pass validation and `model_dump(mode="json")` and still fail the
final UTF-8 encode: a lone surrogate (`os.fsdecode()` of a non-UTF-8 name,
`errors="surrogateescape"` input, decoded `argv`/env values) dies only in
`model_dump_json()` inside a transport writer task, where the failure used to
end the writer and, with it, the session. One unserializable message must cost
only itself, in the same contained-failure shape as the other values that
cannot be framed (`Internal server error`):

- a response or error with an id goes out as `Internal server error` for that id;
- a request is answered locally with `Internal server error` for its id, so the
  awaiting caller fails cleanly instead of waiting forever;
- a notification or id-less error is dropped (logged).

Escaping the surrogate is not an alternative: the peer's JSON parser rejects
lone surrogates in hex escapes ("lone leading surrogate in hex escape"), so the
escaped frame would fail on the far side anyway, and substituting a replacement
character would silently corrupt the very data (file names) the message carries.
"""

from __future__ import annotations

import logging

import anyio
from mcp_types import (
    INTERNAL_ERROR,
    ErrorData,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestId,
)

from mcp.shared._stream_protocols import WriteStream
from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)


def serialize_message(message: JSONRPCMessage) -> str:
    """The wire frame for a message: one JSON document.

    This is the framing the transports have always written
    (`model_dump_json(by_alias=True, exclude_unset=True)`).
    """
    return message.model_dump_json(by_alias=True, exclude_unset=True)


def serialize_outbound_message(message: JSONRPCMessage) -> str | None:
    """Serialize one outbound message, containing a serialization failure to it.

    Returns the wire frame, or `None` when this message must not go on the wire
    (see the module docstring): the writer then routes the failure with
    `deliver_local_error()` and continues with the next message.
    """
    try:
        return serialize_message(message)
    except Exception:
        logger.exception("Failed to serialize outbound JSON-RPC message; failing only this message")
        return _fallback_frame(message)


async def deliver_local_error(message: JSONRPCMessage, read_stream_writer: WriteStream[SessionMessage]) -> None:
    """Report an unserializable message's failure to whoever holds its id.

    Only requests have a waiter: one is answered locally with `Internal server
    error` for its id. Anything else is silently dropped - the session must keep
    running for the messages that can be delivered.
    """
    if isinstance(message, JSONRPCRequest):
        try:
            await read_stream_writer.send(SessionMessage(_internal_error(message.id)))
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass  # the session is ending; nobody is waiting for this error


def _fallback_frame(message: JSONRPCMessage) -> str | None:
    if isinstance(message, JSONRPCResponse | JSONRPCError) and message.id is not None:
        return serialize_message(_internal_error(message.id))
    return None


def _internal_error(request_id: RequestId) -> JSONRPCError:
    return JSONRPCError(
        jsonrpc="2.0", id=request_id, error=ErrorData(code=INTERNAL_ERROR, message="Internal server error")
    )


__all__ = [
    "deliver_local_error",
    "serialize_message",
    "serialize_outbound_message",
]
