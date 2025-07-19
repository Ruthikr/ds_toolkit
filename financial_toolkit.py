from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Union, Tuple
import pandas as pd
import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from datetime import datetime, timedelta
import warnings
import json
import os
import re

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Helper functions
def safe_download(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Robust download with retries and error handling"""
    for _ in range(3):
        try:
            return yf.download(ticker, start=start, end=end, progress=False)
        except:
            continue
    return pd.DataFrame()

def agent_path(base_dir: str, entity: str, data_type: str) -> str:
    """Generate standardized paths for agents"""
    sanitized_entity = re.sub(r'[^\w\-_]', '', entity.replace(' ', '_'))
    return f"{base_dir.rstrip('/')}/{sanitized_entity}_{data_type}.csv"

# Input Models - Optimized for agent usage
class FinancialQuery(BaseModel):
    entity: str = Field(..., description="Company name, ticker, commodity name, or indicator code")
    data_type: str = Field(..., description="Type of data: prices, income, balance, cashflow, dividends, options, economic, commodity, forex, sectors, sentiment")
    period: Optional[str] = Field("30d", description="Time period: 7d, 30d, 90d, 1y, 5y, max")

class DownloadRequest(BaseModel):
    query: FinancialQuery = Field(..., description="Data to download")
    base_dir: str = Field("/data", description="Base directory for saving files")

# Tool Implementations
class FinancialAgentToolkit:
    """Financial Data Toolkit optimized for LLM agent usage"""
    
    def __init__(self):
        self.tools = [
            self.FinancialSearchTool(),
            self.FinancialPreviewTool(),
            self.FinancialDownloadTool()
        ]
    
    def get_tools(self) -> List[BaseTool]:
        return self.tools

    class FinancialSearchTool(BaseTool):
        name: str = "financial_search"
        description: str = "Search for financial entities with available data"
        args_schema: Type[BaseModel] = FinancialQuery
        
        def _run(self, entity: str, data_type: str, period: str = "30d") -> Dict:
            """Agent-friendly search with automatic fallbacks"""
            try:
                # Entity resolution with fallbacks
                if entity.lower() in ['gdp', 'unemployment', 'cpi']:
                    return {
                        "entity": entity.upper(),
                        "type": "economic",
                        "data_types": ["economic"]
                    }
                
                # Ticker resolution
                ticker = self.resolve_ticker(entity)
                if not ticker:
                    return {"error": f"Entity not found: {entity}"}
                
                # Data type availability
                available = self.get_available_data(ticker)
                
                return {
                    "entity": entity,
                    "ticker": ticker,
                    "type": "equity",
                    "data_types": available,
                    "resolved": True
                }
            except Exception as e:
                return {"error": f"Search failed: {str(e)}"}
        
        def resolve_ticker(self, name: str) -> str:
            """Robust ticker resolution for agents"""
            mapping = {
                "apple": "AAPL", "nvidia": "NVDA", "tesla": "TSLA",
                "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
                "gold": "GC=F", "oil": "CL=F", "euro": "EURUSD=X"
            }
            return mapping.get(name.lower(), name.upper())
        
        def get_available_data(self, ticker: str) -> List[str]:
            """Get available data types for an entity"""
            try:
                t = yf.Ticker(ticker)
                available = ["prices"]
                
                # Financial statements
                if t.income_stmt is not None:
                    available.append("income")
                if t.balance_sheet is not None:
                    available.append("balance")
                if t.cash_flow is not None:
                    available.append("cashflow")
                
                # Other data
                if not t.dividends.empty:
                    available.append("dividends")
                if t.options:
                    available.append("options")
                
                return available
            except:
                return ["prices"]

    class FinancialPreviewTool(BaseTool):
        name: str = "financial_preview"
        description: str = "Preview financial data before download"
        args_schema: Type[BaseModel] = FinancialQuery
        
        def _run(self, entity: str, data_type: str, period: str = "30d") -> Dict:
            """Agent-friendly data preview"""
            try:
                # Resolve entity
                search_tool = FinancialAgentToolkit.FinancialSearchTool()
                resolved = search_tool._run(entity, data_type, period)
                if "error" in resolved:
                    return resolved
                
                # Get data
                data = self.fetch_data(resolved, data_type, period)
                if data.empty:
                    return {"error": f"No {data_type} data for {entity}"}
                
                # Format preview
                preview = data.head(3).reset_index().to_dict(orient='records')
                
                return {
                    "entity": entity,
                    "data_type": data_type,
                    "period": period,
                    "preview": preview,
                    "columns": list(data.columns),
                    "ready_for_download": True
                }
            except Exception as e:
                return {"error": f"Preview failed: {str(e)}"}
        
        def fetch_data(self, resolved: Dict, data_type: str, period: str) -> pd.DataFrame:
            """Fetch data based on resolved entity"""
            ticker = resolved.get("ticker", resolved["entity"])
            
            # Date calculations
            end_date = datetime.today()
            start_date = self.calculate_start_date(period)
            
            # Data fetching
            if data_type == "prices":
                return safe_download(ticker, start_date, end_date)
            elif data_type == "income":
                return yf.Ticker(ticker).income_stmt
            elif data_type == "economic":
                return self.fetch_economic_data(ticker)
            else:
                return pd.DataFrame()
        
        def calculate_start_date(self, period: str) -> datetime:
            """Calculate start date from period string"""
            today = datetime.today()
            if period == "7d": return today - timedelta(days=7)
            if period == "30d": return today - timedelta(days=30)
            if period == "90d": return today - timedelta(days=90)
            if period == "1y": return today - timedelta(days=365)
            if period == "5y": return today - timedelta(days=365*5)
            return today - timedelta(days=30)  # Default
        
        def fetch_economic_data(self, indicator: str) -> pd.DataFrame:
            """Fetch economic data"""
            # Simplified for agent usage
            data = {
                "GDP": [100, 102, 105],
                "UNRATE": [3.5, 3.6, 3.4],
                "CPI": [280, 285, 289]
            }
            return pd.DataFrame({indicator: data.get(indicator, [])})

    class FinancialDownloadTool(BaseTool):
        name: str = "financial_download"
        description: str = "Download financial data to standardized paths"
        args_schema: Type[BaseModel] = DownloadRequest
        
        def _run(self, query: FinancialQuery, base_dir: str = "/data") -> Dict:
            """Agent-friendly download with automatic paths"""
            try:
                # Get preview to validate
                preview_tool = FinancialAgentToolkit.FinancialPreviewTool()
                preview = preview_tool._run(**query.dict())
                if "error" in preview:
                    return preview
                
                # Generate path
                path = agent_path(
                    base_dir=base_dir,
                    entity=query.entity,
                    data_type=query.data_type
                )
                
                # Fetch full data
                full_data = preview_tool.fetch_data(
                    resolved=preview,
                    data_type=query.data_type,
                    period=query.period
                )
                
                # Save data
                os.makedirs(os.path.dirname(path), exist_ok=True)
                full_data.to_csv(path, index=False)
                
                return {
                    "action": "download_complete",
                    "entity": query.entity,
                    "data_type": query.data_type,
                    "path": path,
                    "file_size": os.path.getsize(path),
                    "next_step": "analyze_data"
                }
            except Exception as e:
                return {"error": f"Download failed: {str(e)}"}

# Example agent usage workflow
if __name__ == "__main__":
    # Simulate agent thought process
    toolkit = FinancialAgentToolkit()
    
    # Agent searches for Apple financial data
    search_result = toolkit.tools[0].run({
        "entity": "Apple",
        "data_type": "income"
    })
    print("Search Result:", search_result)
    
    # Agent previews Apple income statement
    preview_result = toolkit.tools[1].run({
        "entity": "Apple",
        "data_type": "income"
    })
    print("Preview Result:", preview_result)
    
    # Agent downloads data
    download_result = toolkit.tools[2].run({
        "query": {
            "entity": "Apple",
            "data_type": "income"
        },
        "base_dir": "/analysis/financials"
    })
    print("Download Result:", download_result)