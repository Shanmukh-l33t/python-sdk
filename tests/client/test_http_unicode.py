"""Tests for Unicode handling in streamable HTTP transport.

Verifies that Unicode text is correctly transmitted and received in both directions
(server→client and client→server) using the streamable HTTP transport.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import mcp_types as types
import pytest
from mcp_types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from tests.interaction.transports import StreamingASGITransport

# The in-process app is mounted at this origin purely so URLs are well-formed; nothing listens here.
BASE_URL = "http://127.0.0.1:8000"

# Test constants with various Unicode characters
UNICODE_TEST_STRINGS = {
    "cyrillic": "Слой хранилища, где располагаются",
    "cyrillic_short": "Привет мир",
    "chinese": "你好世界 - 这是一个测试",
    "japanese": "こんにちは世界 - これはテストです",
    "korean": "안녕하세요 세계 - 이것은 테스트입니다",
    "arabic": "مرحبا بالعالم - هذا اختبار",
    "hebrew": "שלום עולם - זה מבחן",
    "greek": "Γεια σου κόσμε - αυτό είναι δοκιμή",
    "emoji": "Hello 👋 World 🌍 - Testing 🧪 Unicode ✨",
    "math": "∑ ∫ √ ∞ ≠ ≤ ≥ ∈ ∉ ⊆ ⊇",
    "accented": "Café, naïve, résumé, piñata, Zürich",
    "mixed": "Hello世界🌍Привет안녕مرحباשלום",
    "special": "Line\nbreak\ttab\r\nCRLF",
    "quotes": '«French» „German" "English" 「Japanese」',
    "currency": "€100 £50 ¥1000 ₹500 ₽200 ¢99",
}


async def handle_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            Tool(
                name="echo_unicode",
                description="🔤 Echo Unicode text - Hello 👋 World 🌍 - Testing 🧪 Unicode ✨",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo back"},
                    },
                    "required": ["text"],
                },
            ),
        ]
    )


async def handle_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
    assert params.name == "echo_unicode"
    assert params.arguments is not None
    return types.CallToolResult(content=[TextContent(type="text", text=f"Echo: {params.arguments['text']}")])


async def handle_list_prompts(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListPromptsResult:
    return types.ListPromptsResult(
        prompts=[
            types.Prompt(
                name="unicode_prompt",
                description="Unicode prompt - Слой хранилища, где располагаются",
                arguments=[],
            )
        ]
    )


async def handle_get_prompt(ctx: ServerRequestContext, params: types.GetPromptRequestParams) -> types.GetPromptResult:
    assert params.name == "unicode_prompt"
    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Hello世界🌍Привет안녕مرحباשלום"),
            )
        ]
    )


@asynccontextmanager
async def unicode_session() -> AsyncIterator[ClientSession]:
    """Yield an initialized ClientSession speaking streamable HTTP (SSE responses) to the
    Unicode test server, entirely in process."""
    server = Server(
        name="unicode_test_server",
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
        on_list_prompts=handle_list_prompts,
        on_get_prompt=handle_get_prompt,
    )
    # SSE response mode, so Unicode rides the SSE event encoding rather than a plain JSON body.
    session_manager = StreamableHTTPSessionManager(app=server, json_response=False)
    app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)])

    async with (
        session_manager.run(),
        # Starlette's Mount 307-redirects the bare /mcp path to /mcp/; the transport follows that
        # same-origin redirect itself.
        httpx2.AsyncClient(transport=StreamingASGITransport(app), base_url=BASE_URL) as http_client,
        streamable_http_client(f"{BASE_URL}/mcp", http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.anyio
async def test_streamable_http_client_unicode_tool_call() -> None:
    """Test that Unicode text is correctly handled in tool calls via streamable HTTP."""
    async with unicode_session() as session:
        # Test 1: List tools (server→client Unicode in descriptions)
        tools = await session.list_tools()
        assert len(tools.tools) == 1

        # Check Unicode in tool descriptions
        echo_tool = tools.tools[0]
        assert echo_tool.name == "echo_unicode"
        assert echo_tool.description is not None
        assert "🔤" in echo_tool.description
        assert "👋" in echo_tool.description

        # Test 2: Send Unicode text in tool call (client→server→client)
        for test_name, test_string in UNICODE_TEST_STRINGS.items():
            result = await session.call_tool("echo_unicode", arguments={"text": test_string})

            # Verify server correctly received and echoed back Unicode
            assert len(result.content) == 1
            content = result.content[0]
            assert content.type == "text"
            assert f"Echo: {test_string}" == content.text, f"Failed for {test_name}"


@pytest.mark.anyio
async def test_streamable_http_client_unicode_prompts() -> None:
    """Test that Unicode text is correctly handled in prompts via streamable HTTP."""
    async with unicode_session() as session:
        # Test 1: List prompts (server→client Unicode in descriptions)
        prompts = await session.list_prompts()
        assert len(prompts.prompts) == 1

        prompt = prompts.prompts[0]
        assert prompt.name == "unicode_prompt"
        assert prompt.description is not None
        assert "Слой хранилища, где располагаются" in prompt.description

        # Test 2: Get prompt with Unicode content (server→client)
        result = await session.get_prompt("unicode_prompt", arguments={})
        assert len(result.messages) == 1

        message = result.messages[0]
        assert message.role == "user"
        assert message.content.type == "text"
        assert message.content.text == "Hello世界🌍Привет안녕مرحباשלום"


# ---------------------------------------------------------------------------
# Outbound serialization failures cost only their own message (#3707)
# ---------------------------------------------------------------------------


def _lone_surrogate() -> str:
    """A str that cannot be UTF-8 encoded: the shape os.fsdecode() returns for a
    non-UTF-8 file name (issue #3707's repro corpus)."""
    return os.fsdecode(b"report-\xff.txt")


async def _handle_surrogate_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            Tool(
                name="bad_result",
                description="returns a result that cannot be serialized",
                input_schema={"type": "object"},
            ),
            Tool(
                name="echo",
                description="echoes its argument",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            ),
        ]
    )


async def _handle_surrogate_call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    if params.name == "bad_result":
        return types.CallToolResult(content=[TextContent(type="text", text=_lone_surrogate())])
    if params.name == "bad_log":
        await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
            level="warning", data={"text": _lone_surrogate()}, related_request_id=ctx.request_id
        )
        return types.CallToolResult(content=[TextContent(type="text", text="logged")])
    assert params.arguments is not None
    return types.CallToolResult(content=[TextContent(type="text", text=f"Echo: {params.arguments['text']}")])


@asynccontextmanager
async def _surrogate_session() -> AsyncIterator[ClientSession]:
    """Yield an initialized ClientSession speaking streamable HTTP (SSE responses) to a
    server that can emit and echo lone surrogates, entirely in process."""
    server = Server(
        name="surrogate_test_server",
        on_list_tools=_handle_surrogate_list_tools,
        on_call_tool=_handle_surrogate_call_tool,
    )
    session_manager = StreamableHTTPSessionManager(app=server, json_response=False)
    app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)])

    async with (
        session_manager.run(),
        httpx2.AsyncClient(transport=StreamingASGITransport(app), base_url=BASE_URL) as http_client,
        streamable_http_client(f"{BASE_URL}/mcp", http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.anyio
async def test_a_surrogate_in_a_tool_result_costs_only_that_call() -> None:
    """A server->client result that cannot be serialized fails that one call with
    Internal server error; the session keeps serving later calls (#3707)."""
    async with _surrogate_session() as session:
        with pytest.raises(MCPError) as exc_info:
            await session.call_tool("bad_result", {}, read_timeout_seconds=5)
        assert exc_info.value.error.code == types.INTERNAL_ERROR
        assert exc_info.value.error.message == "Internal server error"

        result = await session.call_tool("echo", {"text": "hi"}, read_timeout_seconds=5)
        content = result.content[0]
        assert content.type == "text"
        assert content.text == "Echo: hi"


@pytest.mark.anyio
async def test_a_surrogate_in_a_tool_argument_costs_only_that_call() -> None:
    """A client->server argument that cannot be serialized fails that one call with
    Internal server error; the session keeps serving later calls (#3707)."""
    async with _surrogate_session() as session:
        with pytest.raises(MCPError) as exc_info:
            await session.call_tool("echo", {"text": _lone_surrogate()}, read_timeout_seconds=5)
        assert exc_info.value.error.code == types.INTERNAL_ERROR
        assert exc_info.value.error.message == "Internal server error"

        result = await session.call_tool("echo", {"text": "hi"}, read_timeout_seconds=5)
        content = result.content[0]
        assert content.type == "text"
        assert content.text == "Echo: hi"


@pytest.mark.anyio
async def test_a_surrogate_in_a_log_notification_costs_only_that_message() -> None:
    """A server->client notification that cannot be serialized is dropped; the
    call it rode on still completes and later calls still work (#3707)."""
    async with _surrogate_session() as session:
        result = await session.call_tool("bad_log", {}, read_timeout_seconds=5)
        content = result.content[0]
        assert content.type == "text"
        assert content.text == "logged"

        result = await session.call_tool("echo", {"text": "hi"}, read_timeout_seconds=5)
        content = result.content[0]
        assert content.type == "text"
        assert content.text == "Echo: hi"
