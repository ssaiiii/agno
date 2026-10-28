import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agno.tools.optimistfi import OptimistFiTools

ENDPOINT = "https://mcp.optimistfi.com/mcp"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def sse(message: dict) -> str:
    return f"event: message\ndata: {json.dumps(message)}\n\n"


def tool_reply(structured=None, text: str = "", is_error: bool = False) -> str:
    result: dict = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured is not None:
        result["structuredContent"] = structured
    return sse({"jsonrpc": "2.0", "id": 1, "result": result})


def mock_post(mock_client_class, body: str):
    response = MagicMock()
    response.text = body
    response.raise_for_status.return_value = None
    client = mock_client_class.return_value.__enter__.return_value
    client.post.return_value = response
    return client


def sent_call(client) -> dict:
    return client.post.call_args.kwargs["json"]["params"]


def test_initialization_registers_sync_and_async_tools():
    tools = OptimistFiTools()

    assert tools.name == "optimistfi_tools"
    assert set(tools.functions) == {
        "get_perp",
        "get_perp_history",
        "get_perps_market",
        "get_prop_firms",
        "get_insider_trades",
        "get_congress_trades",
        "get_company_filings",
        "get_company_calendar",
    }
    assert set(tools.async_functions) == set(tools.functions)


def test_disabled_tool_is_not_registered():
    tools = OptimistFiTools(enable_get_prop_firms=False)

    assert "get_prop_firms" not in tools.functions
    assert "get_perp" in tools.functions


@patch("agno.tools.optimistfi.httpx.Client")
def test_get_perp_sends_a_tools_call_and_returns_structured_content(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": [{"type": "perp", "entity": "BTC"}]}))
    tools = OptimistFiTools()

    result = json.loads(tools.get_perp("BTC"))

    assert result == {"result": [{"type": "perp", "entity": "BTC"}]}
    client.post.assert_called_once_with(
        ENDPOINT,
        headers=HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_perps", "arguments": {"entity": "BTC"}},
        },
    )


@patch("agno.tools.optimistfi.httpx.Client")
def test_unset_arguments_are_not_sent(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": []}))
    tools = OptimistFiTools()

    tools.get_prop_firms(account_usd=50000)

    assert sent_call(client) == {"name": "get_prop_firms", "arguments": {"account_usd": 50000}}


@patch("agno.tools.optimistfi.httpx.Client")
def test_arguments_map_to_the_server_names(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": []}))
    tools = OptimistFiTools()

    tools.get_insider_trades("NVDA", start_date="2026-09-01", end_date="2026-09-30", limit=5)
    assert sent_call(client) == {
        "name": "get_insider_trades",
        "arguments": {"entity": "NVDA", "start": "2026-09-01", "end": "2026-09-30", "limit": 5},
    }

    tools.get_company_filings("AAPL", kind="red_flags")
    assert sent_call(client) == {
        "name": "get_filings",
        "arguments": {"entity": "AAPL", "kind": "red_flags", "limit": 12},
    }

    tools.get_company_calendar("AAPL")
    assert sent_call(client) == {"name": "get_calendar", "arguments": {"entity": "AAPL", "limit": 12}}


@patch("agno.tools.optimistfi.httpx.Client")
def test_perps_market_drops_empty_filters_and_clamps_limit(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": []}))
    tools = OptimistFiTools()

    tools.get_perps_market(view="lean", market="", venue="", limit=500)

    assert sent_call(client)["arguments"] == {"view": "lean", "limit": 25}


@patch("agno.tools.optimistfi.httpx.Client")
def test_a_filter_the_view_does_not_take_is_left_out_and_named(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": [{"view": "lean"}]}))
    tools = OptimistFiTools()

    result = json.loads(tools.get_perps_market(view="lean", market="BTC", venue="hyperliquid"))

    assert sent_call(client)["arguments"] == {"view": "lean", "market": "BTC", "limit": 10}
    assert result["note"] == "The lean view does not take a venue filter, so it was left out."
    assert result["result"] == [{"view": "lean"}]


@patch("agno.tools.optimistfi.httpx.Client")
def test_options_without_a_market_is_explained_without_a_call(mock_client_class):
    client = mock_client_class.return_value.__enter__.return_value
    tools = OptimistFiTools()

    result = json.loads(tools.get_perps_market(view="options"))

    client.post.assert_not_called()
    assert result == {"error": "The options view reads one market at a time, such as BTC or ETH."}


def test_results_are_json_strings_for_the_model():
    with patch("agno.tools.optimistfi.httpx.Client") as mock_client_class:
        mock_post(mock_client_class, tool_reply({"result": [{"funding": None, "ok": True}]}))
        out = OptimistFiTools().get_perp("BTC")

    assert isinstance(out, str)
    assert json.loads(out) == {"result": [{"funding": None, "ok": True}]}
    assert "None" not in out and "True" not in out


@patch("agno.tools.optimistfi.httpx.Client")
def test_tool_error_returns_the_server_message(mock_client_class):
    mock_post(
        mock_client_class, tool_reply(text="get_perps_history was not run: window: Input should be '1d'", is_error=True)
    )
    tools = OptimistFiTools()

    result = json.loads(tools.get_perp_history("BTC"))

    assert result == {"error": "get_perps_history was not run: window: Input should be '1d'"}


@patch("agno.tools.optimistfi.httpx.Client")
def test_json_rpc_error_is_reported(mock_client_class):
    body = sse({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid params"}})
    mock_post(mock_client_class, body)
    tools = OptimistFiTools()

    result = json.loads(tools.get_perp("BTC"))

    assert result == {"error": "OptimistFi request failed", "detail": "Invalid params"}


@patch("agno.tools.optimistfi.httpx.Client")
def test_text_result_without_structured_content_is_decoded(mock_client_class):
    mock_post(mock_client_class, tool_reply(text='{"entity": "AAPL"}'))
    tools = OptimistFiTools()

    assert json.loads(tools.get_company_calendar("AAPL")) == {"result": {"entity": "AAPL"}}


@patch("agno.tools.optimistfi.httpx.Client")
def test_one_record_per_text_part_is_returned_as_a_list(mock_client_class):
    result = {
        "content": [
            {"type": "text", "text": '{"type": "red_flag", "n": 1}'},
            {"type": "text", "text": '{"type": "red_flag", "n": 2}'},
        ],
        "isError": False,
    }
    mock_post(mock_client_class, sse({"jsonrpc": "2.0", "id": 1, "result": result}))
    tools = OptimistFiTools()

    assert json.loads(tools.get_company_filings("AAPL", kind="red_flags")) == {
        "result": [{"type": "red_flag", "n": 1}, {"type": "red_flag", "n": 2}]
    }


@patch("agno.tools.optimistfi.httpx.Client")
def test_reply_is_found_among_other_stream_messages(mock_client_class):
    notice = sse({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
    mock_post(mock_client_class, notice + tool_reply({"result": [{"ok": True}]}) + notice)
    tools = OptimistFiTools()

    assert json.loads(tools.get_perp("ETH")) == {"result": [{"ok": True}]}


@patch("agno.tools.optimistfi.httpx.Client")
def test_unreadable_body_returns_an_error(mock_client_class):
    mock_post(mock_client_class, "data: not json\n\n")
    tools = OptimistFiTools()

    assert json.loads(tools.get_perp("BTC")) == {"error": "OptimistFi returned a response that is not JSON"}


@patch("agno.tools.optimistfi.httpx.Client")
def test_http_error_preserves_status(mock_client_class):
    request = httpx.Request("POST", ENDPOINT)
    response = MagicMock(spec=httpx.Response)
    response.status_code = 503
    response.text = "error code: 1102"
    client = mock_client_class.return_value.__enter__.return_value
    client.post.return_value = response
    response.raise_for_status.side_effect = httpx.HTTPStatusError("unavailable", request=request, response=response)
    tools = OptimistFiTools()

    result = json.loads(tools.get_perp("BTC"))

    assert result == {"error": "OptimistFi request failed", "status_code": 503, "detail": "error code: 1102"}


@patch("agno.tools.optimistfi.httpx.Client")
def test_network_error_is_reported(mock_client_class):
    client = mock_client_class.return_value.__enter__.return_value
    client.post.side_effect = httpx.ConnectError("connection refused")
    tools = OptimistFiTools()

    result = json.loads(tools.get_perp("BTC"))

    assert result == {"error": "OptimistFi request failed", "detail": "connection refused"}


@pytest.mark.asyncio
@patch("agno.tools.optimistfi.httpx.AsyncClient")
async def test_async_tool_uses_the_same_contract(mock_client_class):
    response = MagicMock()
    response.text = tool_reply({"result": [{"view": "lean"}]})
    response.raise_for_status.return_value = None
    client = mock_client_class.return_value.__aenter__.return_value
    client.post = AsyncMock(return_value=response)
    tools = OptimistFiTools()

    result = json.loads(await tools.aget_perps_market(view="lean", market="BTC"))

    assert result == {"result": [{"view": "lean"}]}
    client.post.assert_awaited_once_with(
        ENDPOINT,
        headers=HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_perps_market", "arguments": {"view": "lean", "market": "BTC", "limit": 10}},
        },
    )


@patch("agno.tools.optimistfi.httpx.Client")
def test_a_market_on_a_view_that_ranks_every_market_is_left_out(mock_client_class):
    client = mock_post(mock_client_class, tool_reply({"result": [{"view": "funding"}]}))
    tools = OptimistFiTools()

    result = json.loads(tools.get_perps_market(view="funding", market="BTC"))

    assert sent_call(client)["arguments"] == {"view": "funding", "limit": 10}
    assert result["note"] == "The funding view does not take a market filter, so it was left out."
