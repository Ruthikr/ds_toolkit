from typing import List, Optional, ClassVar, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import os
import requests
import pandas as pd
import json
import ssl
import certifi
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DOWNLOAD_DIR = "web_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Google API key and CSE ID must be set as environment variables")

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for information or datasets")
    max_results: int = Field(5, description="Max results to return")
    file_types: Optional[str] = Field(None, description="Comma-separated file types to filter by (e.g. 'csv,json'). Omit for general search.")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for information or files. "
        "Returns search results with titles, URLs, and snippets. "
        "Specify file_types parameter to filter for specific file types."
    )
    args_schema: ClassVar[Type[BaseModel]] = WebSearchInput

    def _run(self, query: str, max_results: int = 5, file_types: str = None) -> str:
        try:
            endpoint = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": max_results,
            }
            
            # Add filetype filter if requested
            if file_types:
                params["fileType"] = file_types
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            results = response.json()
            
            formatted_results = []
            for item in results.get("items", [])[:max_results]:
                # Only apply file type filtering if specifically requested
                if file_types:
                    link = item.get("link", "")
                    if not link:
                        continue
                    parsed = urlparse(link)
                    path = parsed.path.lower()
                    if not any(path.endswith(f".{ext.strip()}") for ext in file_types.split(",")):
                        continue
                
                formatted_results.append(
                    f"Title: {item.get('title', 'No title')}\n"
                    f"URL: {item.get('link', '')}\n"
                    f"Snippet: {item.get('snippet', 'No description available')}"
                )
            
            if not formatted_results:
                return "No results found. Try adjusting your search parameters."
            return "\n\n".join(formatted_results)
                
        except Exception as e:
            return f"Search failed: {str(e)}"


class WebDownloadInput(BaseModel):
    url: str = Field(..., description="Direct URL to a file")
    save_dir: Optional[str] = Field(DOWNLOAD_DIR, description="Directory to save the file in")

class WebDownloadTool(BaseTool):
    name: str = "web_download_file"
    description: str = "Download a file from a URL and save locally. Returns the filename."
    args_schema: ClassVar[Type[BaseModel]] = WebDownloadInput

    def _run(self, url: str, save_dir: str = DOWNLOAD_DIR) -> str:
        try:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Clean filename from URL
            filename = url.split("/")[-1].split("?")[0]
            path = os.path.join(save_dir, filename)
            
            # Download with headers to avoid blocking
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Use session with retries and custom SSL context
            session = requests.Session()
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            
            response = session.get(
                url, 
                headers=headers, 
                timeout=30,
                verify=certifi.where(),  # Explicitly use certifi CA bundle
            )

            if response.status_code != 200:
                return f"Download failed: HTTP {response.status_code} - {response.reason}"

            # Save file
            with open(path, "wb") as f:
                f.write(response.content)

            return filename  # Return only the filename
        except requests.exceptions.SSLError:
            # Fallback for sites with SSL issues
            try:
                response = requests.get(url, headers=headers, timeout=30, verify=False)
                if response.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(response.content)
                    return filename  # Return only the filename
                return f"Download failed: HTTP {response.status_code} - {response.reason}"
            except Exception as e:
                return f"SSL fallback failed: {str(e)}"
        except Exception as e:
            return f"Download failed: {str(e)}"


class WebPreviewInput(BaseModel):
    filename: str = Field(..., description="Filename of the downloaded file")
    directory: str = Field(DOWNLOAD_DIR, description="Directory where the file is located")
    rows: int = Field(5, description="How many rows/lines to preview")

class WebPreviewTool(BaseTool):
    name: str = "web_preview_file"
    description: str = "Preview the first few rows of a downloaded CSV or JSON file"
    args_schema: ClassVar[Type[BaseModel]] = WebPreviewInput

    def _run(self, filename: str, directory: str = DOWNLOAD_DIR, rows: int = 5) -> str:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return f"File not found: {path}"
        try:
            if filename.lower().endswith(".csv"):
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(path, on_bad_lines="skip", nrows=1000, encoding=encoding)
                        return df.head(rows).to_string(index=False)
                    except UnicodeDecodeError:
                        continue
                return "Failed to decode CSV with common encodings"
                
            elif filename.lower().endswith(".json"):
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return json.dumps(data[:rows], indent=2)
                    elif isinstance(data, dict):
                        # Handle JSON objects with nested data
                        sample = {k: data[k] for k in list(data.keys())[:2]}
                        return json.dumps(sample, indent=2)
                    else:
                        return str(data)[:500]  # Truncate if too large
            else:
                return "Unsupported format. Only .csv or .json can be previewed."
        except Exception as e:
            return f"Preview failed: {str(e)}"


class WebSearchToolkit:
    def __init__(self):
        self.tools = [
            WebSearchTool(),
            WebDownloadTool(),
            WebPreviewTool()
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools


# Enhanced testing
if __name__ == "__main__":
    toolkit = WebSearchToolkit()
    tools = toolkit.get_tools()

    # Test informational search
    print("\n🔍 Informational Search Test")
    info_results = tools[0].invoke({
        "query": "open-source emotion dataset therapy nlp",
        "max_results": 3
    })
    print(info_results)
    
    # Test dataset search
    print("\n🔍 Dataset Search Test")
    dataset_results = tools[0].invoke({
        "query": "emotion dataset therapy", 
        "max_results": 2,
        "file_types": "csv,json"
    })
    print(dataset_results)
    
    # Test download if dataset found
    if "URL:" in dataset_results:
        first_url = dataset_results.split("URL: ")[1].split("\n")[0]
        print(f"\n⬇️ Downloading: {first_url}")
        download_result = tools[1].invoke({"url": first_url})
        print(f"Download result: {download_result}")
        
        if not download_result.startswith("Download failed"):
            print(f"\n👀 Previewing: {download_result}")
            preview_result = tools[2].invoke({
                "filename": download_result, 
                "rows": 3
            })
            print(preview_result)
    
    # Test mixed result search
    print("\n🔍 Mixed Search Test")
    mixed_results = tools[0].invoke({
        "query": "mental health data resources",
        "max_results": 3
    })
    print(mixed_results)