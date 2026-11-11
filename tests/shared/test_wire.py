"""Outbound wire serialization contains a per-message failure (#3707).

A lone surrogate (`os.fsdecode()` of a non-UTF-8 name) passes validation and
`model_dump(mode="json")` but cannot be UTF-8 encoded: `serialize_message` raises
for it, and the containment layer turns that into one failed message instead of
a dead writer.
"""

import json
import os

import anyio
import pytest
from mcp_types import (
    INTERNAL_ERROR,
    ErrorData,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    jsonrpc_message_adapter,
)

from mcp.shared.message import SessionMessage
from mcp.shared.wire import deliver_local_error, serialize_message, serialize_outbound_message


def _lone_surrogate() -> str:
    return os.fsdecode(b"report-\xff.txt")


def _internal_error_for(request_id: int) -> JSONRPCError:
    return JSONRPCError(
        jsonrpc="2.0", id=request_id, error=ErrorData(code=INTERNAL_ERROR, message="Internal server error")
    )


def test_serialize_message_is_the_transports_frame() -> None:
    message = JSONRPCResponse(jsonrpc="2.0", id=1, result={"ok": True})
    assert serialize_message(message) == message.model_dump_json(by_alias=True, exclude_unset=True)


def test_serialize_message_raises_for_a_lone_surrogate() -> None:
    message = JSONRPCResponse(jsonrpc="2.0", id=1, result={"text": _lone_surrogate()})
    with pytest.raises(Exception):
        serialize_message(message)


def test_a_serializable_message_passes_through() -> None:
    message = JSONRPCRequest(jsonrpc="2.0", id=1, method="ping")
    assert serialize_outbound_message(message) == serialize_message(message)


def test_an_unserializable_response_becomes_internal_error_for_its_id() -> None:
    message = JSONRPCResponse(jsonrpc="2.0", id=7, result={"text": _lone_surrogate()})
    frame = serialize_outbound_message(message)
    assert frame is not None
    assert jsonrpc_message_adapter.validate_json(frame) == _internal_error_for(7)


def test_an_unserializable_error_response_becomes_internal_error_for_its_id() -> None:
    message = JSONRPCError(
        jsonrpc="2.0", id=8, error=ErrorData(code=-1, message="boom", data={"text": _lone_surrogate()})
    )
    frame = serialize_outbound_message(message)
    assert frame is not None
    assert jsonrpc_message_adapter.validate_json(frame) == _internal_error_for(8)


def test_an_unserializable_id_less_error_is_dropped() -> None:
    message = JSONRPCError(
        jsonrpc="2.0", id=None, error=ErrorData(code=-1, message="boom", data={"text": _lone_surrogate()})
    )
    assert serialize_outbound_message(message) is None


def test_an_unserializable_request_is_not_written() -> None:
    message = JSONRPCRequest(jsonrpc="2.0", id=3, method="tools/call", params={"text": _lone_surrogate()})
    assert serialize_outbound_message(message) is None


def test_an_unserializable_notification_is_dropped() -> None:
    message = JSONRPCNotification(jsonrpc="2.0", method="notifications/message", params={"text": _lone_surrogate()})
    assert serialize_outbound_message(message) is None


@pytest.mark.anyio
async def test_deliver_local_error_answers_a_request_with_internal_error() -> None:
    message = JSONRPCRequest(jsonrpc="2.0", id=7, method="elicitation/create", params={"text": _lone_surrogate()})
    send_stream, receive_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    async with receive_stream, send_stream:
        await deliver_local_error(message, send_stream)

        delivered = await receive_stream.receive()
    assert isinstance(delivered, SessionMessage)
    assert delivered.message == _internal_error_for(7)
    assert json.loads(serialize_message(delivered.message))["error"] == {
        "code": INTERNAL_ERROR,
        "message": "Internal server error",
    }


@pytest.mark.anyio
async def test_deliver_local_error_drops_a_notification() -> None:
    message = JSONRPCNotification(jsonrpc="2.0", method="notifications/message", params={"text": _lone_surrogate()})
    send_stream, receive_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    async with receive_stream, send_stream:
        await deliver_local_error(message, send_stream)

        # Nothing was sent: the buffer stays empty.
        assert send_stream.statistics().current_buffer_used == 0
        assert receive_stream.statistics().current_buffer_used == 0


@pytest.mark.anyio
async def test_deliver_local_error_tolerates_a_closing_session() -> None:
    message = JSONRPCRequest(jsonrpc="2.0", id=7, method="elicitation/create", params={"text": _lone_surrogate()})
    send_stream, receive_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    async with receive_stream, send_stream:
        await receive_stream.aclose()

        # The session is ending and nobody waits for the error: no exception escapes.
        await deliver_local_error(message, send_stream)
