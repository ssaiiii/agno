"""
OptimistFi Tools
================

Demonstrates perpetual futures research with OptimistFi: funding, open interest,
liquidation levels and big-account positioning across venues, crypto prop firm
rules and prices, and US company insider trades and filings from SEC EDGAR.

No API key is needed.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.optimistfi import OptimistFiTools

agent = Agent(
    name="Perps Research Agent",
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[OptimistFiTools()],
    instructions=[
        "Name the venue and the time of each figure you quote.",
        "Treat the data as research context, not as trading advice.",
    ],
    markdown=True,
)


if __name__ == "__main__":
    agent.print_response(
        "Where is BTC funding highest right now, which way are Hyperliquid's largest accounts leaning, "
        "and where would the most BTC positions be liquidated?"
    )
