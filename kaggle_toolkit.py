import subprocess
import os
import tempfile
import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# ------------------- CLI Runner -------------------

def run_kaggle_command(command: List[str]) -> str:
    try:
        env = os.environ.copy()
        env["KAGGLE_PAGER"] = ""
        result = subprocess.run(
            ["kaggle"] + command,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise ValueError(f"Kaggle CLI command timed out: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        error_msg = (e.stdout + e.stderr).strip() or "Unknown error"
        raise ValueError(f"Kaggle CLI error ({' '.join(command)}): {error_msg}")
    except FileNotFoundError:
        raise ValueError("Kaggle CLI not found. Please install it and configure API credentials.")


# ------------------- Input Schemas -------------------

class SearchInput(BaseModel):
    query: str = Field(..., description="Main keyword to search datasets")
    sort_by: Optional[str] = Field(
        "hottest",
        description="Sort order: 'hottest' (default), 'votes', 'updated', 'active'"
    )
    max_results: Optional[int] = Field(
        5,
        description="Limit results between 1 and 50 (default is 5)"
    )


class InfoInput(BaseModel):
    dataset_ref: str = Field(
        ...,
        description="Dataset reference in format: 'owner-slug/dataset-name' (e.g., 'zynicide/wine-reviews')"
    )


class DownloadInput(BaseModel):
    dataset_ref: str = Field(..., description="Dataset reference like 'zynicide/wine-reviews'")
    path: Optional[str] = Field(
        "./kaggle_downloads",
        description="Local directory to store the downloaded dataset"
    )
    unzip: Optional[bool] = Field(
        True,
        description="Unzip the dataset after download (default = true)"
    )


# ------------------- Base Robust Tool -------------------

class RobustTool(BaseTool):
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run(self, tool_input: Union[str, Dict], **kwargs: Any) -> str:
        if isinstance(tool_input, str):
            try:
                parsed = json.loads(tool_input)
                return super().run(parsed, **kwargs)
            except json.JSONDecodeError:
                return self._run_from_string(tool_input)
        return super().run(tool_input, **kwargs)

    def _run_from_string(self, tool_input: str) -> str:
        raise NotImplementedError("This tool doesn't support raw string input")


# ------------------- Kaggle Search Tool -------------------

class KaggleSearchCLITool(RobustTool):
    name: str = "kaggle_search_cli"
    description: str = (
        "🔍 Search for Kaggle datasets using a keyword.\n\n"
        "✅ Accepts either:\n"
        "  • Raw string (e.g., 'credit card fraud')\n"
        "  • JSON (e.g., {\"query\": \"laptop\", \"sort_by\": \"votes\", \"max_results\": 10})\n\n"
        "📌 Parameters:\n"
        "  - query (required): Main keyword to search\n"
        "  - sort_by (optional): 'hottest' (default), 'votes', 'updated', 'active'\n"
        "  - max_results (optional): Limit between 1–50 (default 5)\n\n"
        "🚫 Avoid:\n"
        "  - Empty query string\n"
        "  - Invalid sort_by values\n"
        "  - max_results > 50 or < 1"
    )
    args_schema: type = SearchInput

    def _run(self, query: str, sort_by: str = "hottest", max_results: int = 5) -> str:
        sort_by = sort_by.lower()
        if sort_by not in ["hottest", "votes", "updated", "active"]:
            sort_by = "hottest"

        max_results = max(1, min(max_results, 50))
        try:
            command = ["datasets", "list", "-s", query, "--sort-by", sort_by]
            output = run_kaggle_command(command)
            lines = output.splitlines()
            if len(lines) <= 1:
                return f"No datasets found for '{query}'. Try a broader keyword."

            header = lines[0]
            rows = lines[1:1+max_results]
            total_count = len(lines) - 1

            return (
                f"Top {len(rows)}/{total_count} results for '{query}' "
                f"(sorted by '{sort_by}'):\n\n{header}\n" + "\n".join(rows)
            )
        except ValueError as e:
            return f"Search failed: {str(e)}"

    def _run_from_string(self, tool_input: str) -> str:
        parts = [p.strip() for p in tool_input.split(",")]
        query = parts[0]
        sort_by = "hottest"
        max_results = 5

        for param in parts[1:]:
            if "=" in param:
                key, value = param.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "sort_by":
                    sort_by = value
                elif key == "max_results":
                    try:
                        max_results = int(value)
                    except ValueError:
                        pass
        return self._run(query, sort_by, max_results)


# ------------------- Kaggle Info Tool -------------------

class KaggleInfoCLITool(RobustTool):
    name: str = "kaggle_info_cli"
    description: str = (
        "ℹ️ Fetch metadata for a specific Kaggle dataset.\n\n"
        "✅ Accepts either:\n"
        "  • Raw dataset ref (e.g., 'zynicide/wine-reviews')\n"
        "  • JSON (e.g., {\"dataset_ref\": \"zynicide/wine-reviews\"})\n\n"
        "📌 Required:\n"
        "  - dataset_ref: in the format 'owner/dataset-name'\n\n"
        "🚫 Avoid:\n"
        "  - Misspelled or non-existing dataset references"
    )
    args_schema: type = InfoInput

    def _run(self, dataset_ref: str) -> str:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                command = ["datasets", "metadata", "-d", dataset_ref, "-p", temp_dir]
                run_kaggle_command(command)
                metadata_path = os.path.join(temp_dir, "dataset-metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        return f.read()
                return f"Metadata file not found for: {dataset_ref}"
        except ValueError as e:
            return f"Metadata retrieval failed: {str(e)}"

    def _run_from_string(self, tool_input: str) -> str:
        return self._run(tool_input.strip())


# ------------------- Kaggle Download Tool -------------------

class KaggleDownloadCLITool(RobustTool):
    name: str = "kaggle_download_cli"
    description: str = (
        "⬇️ Download a dataset from Kaggle and optionally unzip it.\n\n"
        "✅ Accepts either:\n"
        "  • Raw string (e.g., 'zynicide/wine-reviews, path=./data, unzip=false')\n"
        "  • JSON (e.g., {\"dataset_ref\": \"zynicide/wine-reviews\", \"path\": \"./data\", \"unzip\": true})\n\n"
        "📌 Parameters:\n"
        "  - dataset_ref (required): in format 'owner/dataset'\n"
        "  - path (optional): directory to save (default './kaggle_downloads')\n"
        "  - unzip (optional): unzip after download (true by default)\n\n"
        "🚫 Avoid:\n"
        "  - Invalid dataset_ref\n"
        "  - Non-writable download path\n"
        "  - Typos like 'unzip=tru' or wrong booleans"
    )
    args_schema: type = DownloadInput

    def _run(self, dataset_ref: str, path: str = "./kaggle_downloads", unzip: bool = True) -> str:
        try:
            os.makedirs(path, exist_ok=True)
            command = ["datasets", "download", "-d", dataset_ref, "-p", path]
            if unzip:
                command.append("--unzip")
            result = run_kaggle_command(command)
            return f"✅ Download complete. Files saved to: {path}\n\n{result}"
        except ValueError as e:
            return f"Download failed: {str(e)}"
        except OSError as e:
            return f"Filesystem error: {str(e)}"

    def _run_from_string(self, tool_input: str) -> str:
        parts = [p.strip() for p in tool_input.split(",")]
        dataset_ref = parts[0]
        path = "./kaggle_downloads"
        unzip = True

        for param in parts[1:]:
            if "=" in param:
                key, value = param.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "path":
                    path = value
                elif key == "unzip":
                    unzip = value.lower() in ["true", "yes", "1"]
        return self._run(dataset_ref, path, unzip)


# ------------------- Toolkit Loader -------------------

class KaggleToolkit:
    def __init__(self):
        self.tools = [
            KaggleSearchCLITool(),
            KaggleInfoCLITool(),
            KaggleDownloadCLITool()
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools