"""OptimistFi tools for Agno agents: perpetual futures, crypto prop firms and US company disclosures."""

import asyncio
import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from agno.tools import Toolkit

PerpsView = Literal[
    "trending",
    "movers",
    "funding",
    "funding_gaps",
    "open_interest",
    "liquidations",
    "liquidation_max_pain",
    "whales",
    "lean",
    "weekend",
    "new_listings",
    "market_overview",
    "fear_greed",
    "etf",
    "sectors",
    "exchanges",
    "options",
    "basis",
    "premium",
    "btc_cycle",
]
# Which filters each view takes, as the server answered on 2026-10-28. A filter a view does not take fails the
# whole call, so get_perps_market leaves it out and says so.
VENUE_VIEWS = {"open_interest", "whales", "new_listings", "exchanges"}
NO_MARKET_VIEWS = {"trending", "funding", "liquidations"}
NEEDS_MARKET_VIEWS = {"options", "basis"}
PerpsWindow = Literal["1h", "4h", "24h"]
HistoryWindow = Literal["1d", "7d", "30d", "90d"]
AssetClass = Literal["crypto", "equity", "index", "commodity", "fx"]
FilingKind = Literal["events", "press_releases", "event_text", "red_flags"]
PropRule = Literal[
    "static-drawdown",
    "under-50",
    "instant-funding",
    "no-daily-loss-limit",
    "stock-perps",
    "hyperliquid",
    "200k-accounts",
    "us-traders",
    "no-minimum-days",
]
PropAsset = Literal["Crypto", "Stocks", "ETFs", "Indices", "Commodities", "Forex"]


class OptimistFiTools(Toolkit):
    """Tools for perpetual futures, crypto prop firms and US company data from OptimistFi.

    The tools call OptimistFi's public MCP endpoint over HTTP. No API key is needed.
    """

    def __init__(
        self,
        base_url: str = "https://mcp.optimistfi.com/mcp",
        timeout: float = 60.0,
        enable_get_perp: bool = True,
        enable_get_perp_history: bool = True,
        enable_get_perps_market: bool = True,
        enable_get_prop_firms: bool = True,
        enable_get_insider_trades: bool = True,
        enable_get_congress_trades: bool = True,
        enable_get_company_filings: bool = True,
        enable_get_company_calendar: bool = True,
        all: bool = False,
        **kwargs,
    ):
        """Initialize the OptimistFi toolkit.

        Args:
            base_url: OptimistFi MCP endpoint.
            timeout: Per-request timeout in seconds.
            enable_get_perp: Register the single perp market tool.
            enable_get_perp_history: Register the perp market history tool.
            enable_get_perps_market: Register the cross-market perps tool.
            enable_get_prop_firms: Register the prop firm tool.
            enable_get_insider_trades: Register the insider trades tool.
            enable_get_congress_trades: Register the congressional trades tool.
            enable_get_company_filings: Register the company filings tool.
            enable_get_company_calendar: Register the company calendar tool.
            all: Register all tools regardless of individual flags.
        """
        self.base_url = base_url
        self.timeout = httpx.Timeout(timeout)

        tools: List[Any] = []
        async_tools: List[tuple] = []
        if all or enable_get_perp:
            tools.append(self.get_perp)
            async_tools.append((self.aget_perp, "get_perp"))
        if all or enable_get_perp_history:
            tools.append(self.get_perp_history)
            async_tools.append((self.aget_perp_history, "get_perp_history"))
        if all or enable_get_perps_market:
            tools.append(self.get_perps_market)
            async_tools.append((self.aget_perps_market, "get_perps_market"))
        if all or enable_get_prop_firms:
            tools.append(self.get_prop_firms)
            async_tools.append((self.aget_prop_firms, "get_prop_firms"))
        if all or enable_get_insider_trades:
            tools.append(self.get_insider_trades)
            async_tools.append((self.aget_insider_trades, "get_insider_trades"))
        if all or enable_get_congress_trades:
            tools.append(self.get_congress_trades)
            async_tools.append((self.aget_congress_trades, "get_congress_trades"))
        if all or enable_get_company_filings:
            tools.append(self.get_company_filings)
            async_tools.append((self.aget_company_filings, "get_company_filings"))
        if all or enable_get_company_calendar:
            tools.append(self.get_company_calendar)
            async_tools.append((self.aget_company_calendar, "get_company_calendar"))

        name = kwargs.pop("name", "optimistfi_tools")
        super().__init__(name=name, tools=tools, async_tools=async_tools, **kwargs)

    # The endpoint speaks MCP's streamable HTTP transport: a JSON-RPC request in, the reply as a server-sent event.
    _HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    # A gateway status is what the server sends while it restarts; one retry rides out a short gap.
    _GATEWAY = frozenset({502, 503, 504})
    _RETRY_SECONDS = 5.0

    @staticmethod
    def _payload(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": {k: v for k, v in arguments.items() if v not in (None, "")}},
        }

    @staticmethod
    def _parse(body: str) -> Dict[str, Any]:
        messages = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")] or [body]
        reply: Dict[str, Any] = {}
        for message in messages:
            if not message:
                continue
            decoded = json.loads(message)
            if isinstance(decoded, dict) and decoded.get("id") == 1:
                reply = decoded
        if "error" in reply:
            return {"error": "OptimistFi request failed", "detail": reply["error"].get("message")}
        result = reply.get("result")
        if not isinstance(result, dict):
            return {"error": "OptimistFi returned no result"}
        parts = [part.get("text", "") for part in result.get("content", []) if part.get("type") == "text"]
        if result.get("isError"):
            return {"error": " ".join(parts) or "OptimistFi request failed"}
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        # Without structured content, each text part carries one JSON record.
        try:
            records = [json.loads(part) for part in parts]
        except ValueError:
            return {"result": "\n".join(parts)}
        return {"result": records[0] if len(records) == 1 else records}

    @staticmethod
    def _dumps(result: Dict[str, Any]) -> str:
        # Compact JSON: OptimistFi answers run to tens of kilobytes, and indentation would only add tokens.
        return json.dumps(result, ensure_ascii=False)

    def _call(self, tool: str, **arguments: Any) -> str:
        return self._dumps(self._call_dict(tool, **arguments))

    async def _acall(self, tool: str, **arguments: Any) -> str:
        return self._dumps(await self._acall_dict(tool, **arguments))

    def _call_dict(self, tool: str, **arguments: Any) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.base_url, headers=self._HEADERS, json=self._payload(tool, arguments))
                if response.status_code in self._GATEWAY:
                    time.sleep(self._RETRY_SECONDS)
                    response = client.post(self.base_url, headers=self._HEADERS, json=self._payload(tool, arguments))
                response.raise_for_status()
                return self._parse(response.text)
        except httpx.HTTPStatusError as error:
            return self._status_error(error)
        except httpx.RequestError as error:
            return {"error": "OptimistFi request failed", "detail": str(error)}
        except ValueError:
            return {"error": "OptimistFi returned a response that is not JSON"}

    async def _acall_dict(self, tool: str, **arguments: Any) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, headers=self._HEADERS, json=self._payload(tool, arguments))
                if response.status_code in self._GATEWAY:
                    await asyncio.sleep(self._RETRY_SECONDS)
                    response = await client.post(
                        self.base_url, headers=self._HEADERS, json=self._payload(tool, arguments)
                    )
                response.raise_for_status()
                return self._parse(response.text)
        except httpx.HTTPStatusError as error:
            return self._status_error(error)
        except httpx.RequestError as error:
            return {"error": "OptimistFi request failed", "detail": str(error)}
        except ValueError:
            return {"error": "OptimistFi returned a response that is not JSON"}

    @classmethod
    def _status_error(cls, error: httpx.HTTPStatusError) -> Dict[str, Any]:
        code = error.response.status_code
        if code in cls._GATEWAY:
            return {
                "error": "OptimistFi's server did not answer on two tries "
                f"{cls._RETRY_SECONDS:.0f} seconds apart. It sends this status while it restarts, "
                "which takes a few minutes.",
                "status_code": code,
            }
        return {"error": "OptimistFi request failed", "status_code": code, "detail": error.response.text[:500]}

    def get_perp(self, market: str) -> str:
        """Get one perpetual futures market across venues right now: price and premium to the index or stock,
        open interest, funding, maximum leverage, liquidation levels, large trades and positioning. Covers crypto,
        stock, index and commodity perps; a stock perp also shows its move while the stock market is closed.

        Args:
            market: A ticker or symbol (BTC, NVDA, XAU), a company name (Tesla) or an asset name (bitcoin, gold).
        """
        return self._call("get_perps", entity=market)

    async def aget_perp(self, market: str) -> str:
        """Get one perpetual futures market across venues right now: price and premium to the index or stock,
        open interest, funding, maximum leverage, liquidation levels, large trades and positioning. Covers crypto,
        stock, index and commodity perps; a stock perp also shows its move while the stock market is closed.

        Args:
            market: A ticker or symbol (BTC, NVDA, XAU), a company name (Tesla) or an asset name (bitcoin, gold).
        """
        return await self._acall("get_perps", entity=market)

    def get_perp_history(self, market: str, window: HistoryWindow = "7d") -> str:
        """Get one perpetual futures market over time: price, open interest, funding, taker buying against
        selling and the net position of large accounts, plus the latest order book depth.

        Args:
            market: A ticker or symbol (BTC, NVDA, XAU), a company name (Tesla) or an asset name (bitcoin, gold).
            window: How far back: 1d (hourly steps), 7d (4-hourly), 30d or 90d (daily).
        """
        return self._call("get_perps_history", entity=market, window=window)

    async def aget_perp_history(self, market: str, window: HistoryWindow = "7d") -> str:
        """Get one perpetual futures market over time: price, open interest, funding, taker buying against
        selling and the net position of large accounts, plus the latest order book depth.

        Args:
            market: A ticker or symbol (BTC, NVDA, XAU), a company name (Tesla) or an asset name (bitcoin, gold).
            window: How far back: 1d (hourly steps), 7d (4-hourly), 30d or 90d (daily).
        """
        return await self._acall("get_perps_history", entity=market, window=window)

    @staticmethod
    def _market_view_arguments(
        view: str,
        market: Optional[str],
        venue: Optional[str],
        asset_class: Optional[str],
        window: Optional[str],
        limit: int,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        if view in NEEDS_MARKET_VIEWS and not market:
            return None, []
        left_out = []
        if market and view in NO_MARKET_VIEWS:
            left_out.append("market")
            market = None
        if venue and view not in VENUE_VIEWS:
            left_out.append("venue")
            venue = None
        arguments = {
            "view": view,
            "market": market,
            "venue": venue,
            "asset_class": asset_class,
            "window": window,
            "limit": max(1, min(limit, 25)),
        }
        return arguments, left_out

    @staticmethod
    def _with_note(result: Dict[str, Any], view: str, left_out: List[str]) -> Dict[str, Any]:
        if not left_out:
            return result
        named = " and ".join(f"{name} filter" for name in left_out)
        return {"note": f"The {view} view does not take a {named}, so it was left out.", **result}

    def get_perps_market(
        self,
        view: PerpsView = "trending",
        market: Optional[str] = None,
        venue: Optional[str] = None,
        asset_class: Optional[AssetClass] = None,
        window: Optional[PerpsWindow] = None,
        limit: int = 10,
    ) -> str:
        """Get one view across perpetual futures markets and venues.

        Views: trending (where open interest is building), movers (biggest price moves), funding (highest and
        lowest funding rates), funding_gaps (the same market's funding on different venues), open_interest,
        liquidations, liquidation_max_pain (the price band where the most positions would be liquidated, from
        Hyperliquid's mapped positions), whales (trades of $100K or more), lean (long against short for
        Hyperliquid accounts worth $1M or more), weekend (stock perps' moves while US stocks are closed),
        new_listings, market_overview, fear_greed (a 0 to 100 crypto sentiment index), etf (spot bitcoin or ether
        ETF flows and holdings), sectors, exchanges (venues ranked by open interest), options (implied volatility
        and skew by expiry), basis (dated futures' premium to the index), premium (perps furthest from their index)
        and btc_cycle (bitcoin cycle and peak indicators).

        Args:
            view: The view to read.
            market: One market (BTC, NVDA, gold). options and basis need one; trending, funding and liquidations
                rank every market and take none; the other views narrow to it.
            venue: One venue id (hyperliquid, dydx, aster, lighter, gmx). Only whales, new_listings, open_interest
                and exchanges take a venue; lean already covers Hyperliquid alone.
            asset_class: Only this kind of market: crypto, equity, index, commodity or fx.
            window: Look-back for trending, movers, liquidations, whales and new_listings.
            limit: Rows per list, from 1 to 25.
        """
        arguments, left_out = self._market_view_arguments(view, market, venue, asset_class, window, limit)
        if arguments is None:
            return self._dumps({"error": f"The {view} view reads one market at a time, such as BTC or ETH."})
        return self._dumps(self._with_note(self._call_dict("get_perps_market", **arguments), view, left_out))

    async def aget_perps_market(
        self,
        view: PerpsView = "trending",
        market: Optional[str] = None,
        venue: Optional[str] = None,
        asset_class: Optional[AssetClass] = None,
        window: Optional[PerpsWindow] = None,
        limit: int = 10,
    ) -> str:
        """Get one view across perpetual futures markets and venues.

        Views: trending (where open interest is building), movers (biggest price moves), funding (highest and
        lowest funding rates), funding_gaps (the same market's funding on different venues), open_interest,
        liquidations, liquidation_max_pain (the price band where the most positions would be liquidated, from
        Hyperliquid's mapped positions), whales (trades of $100K or more), lean (long against short for
        Hyperliquid accounts worth $1M or more), weekend (stock perps' moves while US stocks are closed),
        new_listings, market_overview, fear_greed (a 0 to 100 crypto sentiment index), etf (spot bitcoin or ether
        ETF flows and holdings), sectors, exchanges (venues ranked by open interest), options (implied volatility
        and skew by expiry), basis (dated futures' premium to the index), premium (perps furthest from their index)
        and btc_cycle (bitcoin cycle and peak indicators).

        Args:
            view: The view to read.
            market: One market (BTC, NVDA, gold). options and basis need one; trending, funding and liquidations
                rank every market and take none; the other views narrow to it.
            venue: One venue id (hyperliquid, dydx, aster, lighter, gmx). Only whales, new_listings, open_interest
                and exchanges take a venue; lean already covers Hyperliquid alone.
            asset_class: Only this kind of market: crypto, equity, index, commodity or fx.
            window: Look-back for trending, movers, liquidations, whales and new_listings.
            limit: Rows per list, from 1 to 25.
        """
        arguments, left_out = self._market_view_arguments(view, market, venue, asset_class, window, limit)
        if arguments is None:
            return self._dumps({"error": f"The {view} view reads one market at a time, such as BTC or ETH."})
        return self._dumps(self._with_note(await self._acall_dict("get_perps_market", **arguments), view, left_out))

    def get_prop_firms(
        self,
        firm: Optional[str] = None,
        account_usd: Optional[int] = None,
        max_price_usd: Optional[float] = None,
        rule: Optional[PropRule] = None,
        asset: Optional[PropAsset] = None,
        country: Optional[str] = None,
        on_chain: Optional[bool] = None,
    ) -> str:
        """Get prop firms that fund crypto and stock perps traders, read from the firms' own pages: rules,
        challenge prices, live discount codes, country, platforms, assets and maximum allocation. Without a filter
        it returns a short record for every firm, which is long; pass a filter where the question allows one.

        Args:
            firm: One firm by name (Breakout, HyroTrader, Propr): its full record, every challenge and its dated
                rule changes.
            account_usd: Only challenges of this account size in dollars (50000), cheapest to get funded first.
            max_price_usd: Only firms whose cheapest evaluation costs this much or less, in dollars.
            rule: Only firms on one rule list, such as no-daily-loss-limit, instant-funding or us-traders.
            asset: Only firms that offer this asset class.
            country: Only firms based in this country.
            on_chain: True for only firms whose trades execute on an on-chain venue.
        """
        return self._call(
            "get_prop_firms",
            firm=firm,
            account_usd=account_usd,
            max_price_usd=max_price_usd,
            rule=rule,
            asset=asset,
            country=country,
            on_chain=on_chain,
        )

    async def aget_prop_firms(
        self,
        firm: Optional[str] = None,
        account_usd: Optional[int] = None,
        max_price_usd: Optional[float] = None,
        rule: Optional[PropRule] = None,
        asset: Optional[PropAsset] = None,
        country: Optional[str] = None,
        on_chain: Optional[bool] = None,
    ) -> str:
        """Get prop firms that fund crypto and stock perps traders, read from the firms' own pages: rules,
        challenge prices, live discount codes, country, platforms, assets and maximum allocation. Without a filter
        it returns a short record for every firm, which is long; pass a filter where the question allows one.

        Args:
            firm: One firm by name (Breakout, HyroTrader, Propr): its full record, every challenge and its dated
                rule changes.
            account_usd: Only challenges of this account size in dollars (50000), cheapest to get funded first.
            max_price_usd: Only firms whose cheapest evaluation costs this much or less, in dollars.
            rule: Only firms on one rule list, such as no-daily-loss-limit, instant-funding or us-traders.
            asset: Only firms that offer this asset class.
            country: Only firms based in this country.
            on_chain: True for only firms whose trades execute on an on-chain venue.
        """
        return await self._acall(
            "get_prop_firms",
            firm=firm,
            account_usd=account_usd,
            max_price_usd=max_price_usd,
            rule=rule,
            asset=asset,
            country=country,
            on_chain=on_chain,
        )

    def get_insider_trades(
        self,
        company: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 25,
    ) -> str:
        """Get a US company's Form 4 insider transactions from SEC EDGAR: who, role, what they did (direction and
        transaction, with the Form 4 code in txn_code), shares and dollar value. Open-market buys and sells come
        first, then grants, option exercises, tax withholding and gifts, newest first within each group, so with a
        small limit a recent gift can sit below older sales. Each record's event_time is the trade date and as_of
        is the filing date, when the trade became public.

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            start_date: Earliest transaction date, YYYY-MM-DD.
            end_date: Latest transaction date, YYYY-MM-DD.
            limit: Most transactions to return.
        """
        return self._call("get_insider_trades", entity=company, start=start_date, end=end_date, limit=limit)

    async def aget_insider_trades(
        self,
        company: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 25,
    ) -> str:
        """Get a US company's Form 4 insider transactions from SEC EDGAR: who, role, what they did (direction and
        transaction, with the Form 4 code in txn_code), shares and dollar value. Open-market buys and sells come
        first, then grants, option exercises, tax withholding and gifts, newest first within each group, so with a
        small limit a recent gift can sit below older sales. Each record's event_time is the trade date and as_of
        is the filing date, when the trade became public.

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            start_date: Earliest transaction date, YYYY-MM-DD.
            end_date: Latest transaction date, YYYY-MM-DD.
            limit: Most transactions to return.
        """
        return await self._acall("get_insider_trades", entity=company, start=start_date, end=end_date, limit=limit)

    def get_congress_trades(self, ticker: Optional[str] = None, member: Optional[str] = None, limit: int = 25) -> str:
        """Get US House stock-trade disclosures filed under the STOCK Act, by ticker or by member, newest first.
        Each record names the account the trade was made in (account), so two records alike in all else are trades
        in two accounts; account_not_read marks one whose account is not read yet. Amounts are the filing's range:
        an exact amount is both bounds, and "Over $X" has amount_high null. event_time is the trade date;
        disclosure_date (and as_of) is the date the filing became public; notification_date is the date the filing
        says the member was notified of the trade.

        Args:
            ticker: A stock ticker: which members traded it.
            member: A House member's name: what that member traded.
            limit: Most trades to return.
        """
        return self._call("get_congress_trades", ticker=ticker, member=member, limit=limit)

    async def aget_congress_trades(
        self, ticker: Optional[str] = None, member: Optional[str] = None, limit: int = 25
    ) -> str:
        """Get US House stock-trade disclosures filed under the STOCK Act, by ticker or by member, newest first.
        Each record names the account the trade was made in (account), so two records alike in all else are trades
        in two accounts; account_not_read marks one whose account is not read yet. Amounts are the filing's range:
        an exact amount is both bounds, and "Over $X" has amount_high null. event_time is the trade date;
        disclosure_date (and as_of) is the date the filing became public; notification_date is the date the filing
        says the member was notified of the trade.

        Args:
            ticker: A stock ticker: which members traded it.
            member: A House member's name: what that member traded.
            limit: Most trades to return.
        """
        return await self._acall("get_congress_trades", ticker=ticker, member=member, limit=limit)

    def get_company_filings(
        self,
        company: str,
        kind: FilingKind = "events",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 12,
    ) -> str:
        """Get a US company's own disclosures from SEC EDGAR: filed 8-K events, its press releases, the text of
        its latest 8-Ks, or governance red flags such as late filings, auditor changes and insider selling clusters.

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            kind: events, press_releases, event_text or red_flags.
            start_date: Window start, YYYY-MM-DD.
            end_date: Window end, YYYY-MM-DD.
            limit: Most records to return, newest first.
        """
        return self._call("get_filings", entity=company, kind=kind, start=start_date, end=end_date, limit=limit)

    async def aget_company_filings(
        self,
        company: str,
        kind: FilingKind = "events",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 12,
    ) -> str:
        """Get a US company's own disclosures from SEC EDGAR: filed 8-K events, its press releases, the text of
        its latest 8-Ks, or governance red flags such as late filings, auditor changes and insider selling clusters.

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            kind: events, press_releases, event_text or red_flags.
            start_date: Window start, YYYY-MM-DD.
            end_date: Window end, YYYY-MM-DD.
            limit: Most records to return, newest first.
        """
        return await self._acall("get_filings", entity=company, kind=kind, start=start_date, end=end_date, limit=limit)

    def get_company_calendar(self, company: str, limit: int = 12) -> str:
        """Get what is ahead for a US company: the next earnings date projected from its own filing cadence, marked
        as an estimate with its basis, first, then clinical trial readouts, nearest first. When more readouts are due
        than the limit holds, later-phase studies are kept; a readout's data.date_precision is "month" when
        ClinicalTrials.gov gives only the month of completion (event_time is then the 1st).

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            limit: Most events to return.
        """
        return self._call("get_calendar", entity=company, limit=limit)

    async def aget_company_calendar(self, company: str, limit: int = 12) -> str:
        """Get what is ahead for a US company: the next earnings date projected from its own filing cadence, marked
        as an estimate with its basis, first, then clinical trial readouts, nearest first. When more readouts are due
        than the limit holds, later-phase studies are kept; a readout's data.date_precision is "month" when
        ClinicalTrials.gov gives only the month of completion (event_time is then the 1st).

        Args:
            company: A ticker (AAPL, BRK.B), a company name (Apple) or an SEC CIK.
            limit: Most events to return.
        """
        return await self._acall("get_calendar", entity=company, limit=limit)
