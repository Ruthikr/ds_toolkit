from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import pandas as pd
import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class StockDataInput(BaseModel):
    input_string: str = Field(..., description="A string containing the ticker, start_date, and end_date, separated by commas. dont pass js9n or dict just pass an string separated by c9mmas for different arguments")


class GetStockDataTool(BaseTool):
    name: str = "get_stock_data"
    description: str = "Fetches historical stock data from Yahoo Finance."
    args_schema: Type[BaseModel] = StockDataInput

    def _run(self, input_string: str) -> pd.DataFrame:
        try:
            parts = [p.strip() for p in input_string.split(',')]
            if len(parts) != 3:
                return "Invalid input string. Please provide the ticker, start_date, and end_date, separated by commas."

            ticker, start_date, end_date = parts
            return yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            return f"Failed to fetch stock data: {e}"


class NewsInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL').")


class GetNewsTool(BaseTool):
    name: str = "get_financial_news"
    description: str = "Fetches financial news for a given stock ticker."
    args_schema: Type[BaseModel] = NewsInput

    def _run(self, ticker: str) -> List[Dict[str, Any]]:
        try:
            return yf.Ticker(ticker).news
        except Exception as e:
            return f"Failed to fetch news: {e}"


class FinancialToolkit:
    """
    Toolkit for fetching financial data.
    """

    def __init__(self):
        self.tools = [
            GetStockDataTool(),
            GetNewsTool(),
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools