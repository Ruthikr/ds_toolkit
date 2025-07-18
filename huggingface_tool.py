
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from huggingface_hub import HfApi
from datasets import load_dataset


api = HfApi()

# ------------------- Input Schemas -------------------

class HFSearchInput(BaseModel):
    query: str = Field(..., description="Keyword to search datasets")
    max_results: int = Field(5, description="Max number of results to return")


class HFInfoInput(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID like 'imdb', 'glue'")


class HFDownloadInput(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID")
    split: Optional[str] = Field(None, description="Optional split like 'train'")
    local_dir: Optional[str] = Field("./hf_datasets", description="Where to save locally")


class HFPreviewInput(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID")
    split: Optional[str] = Field("train", description="Split to preview")
    num_rows: Optional[int] = Field(5, description="How many rows to preview")


# ------------------- Tools -------------------

class HFSearchTool(BaseTool):
    name: str = "hf_search_datasets"
    description: str = "Search public Hugging Face datasets by keyword"
    args_schema: type = HFSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            # Convert the generator to a list to allow for checking its length
            results = list(api.list_datasets(search=query, limit=max_results))
            
            if not results:
                return "No datasets found."
            
            # Return only the ID of the top search result as a clean string.
            return results[0].id.strip()
        except Exception as e:
            return f"Failed to search for datasets: {e}"


class HFInfoTool(BaseTool):
    name: str = "hf_dataset_info"
    description: str = "Get metadata and description of a Hugging Face dataset"
    args_schema: type = HFInfoInput

    def _run(self, dataset_id: str) -> str:
        try:
            info = api.dataset_info(dataset_id)
            # Ensure cardData exists and handle if it's None
            card_data = info.cardData if info.cardData else {}
            return (
                f"ID: {info.id}\n"
                f"Title: {card_data.get('title', 'N/A')}\n"
                f"Description: {getattr(info, 'description', 'N/A')}\n"
                f"Downloads: {getattr(info, 'downloads', 'N/A')}\n"
                f"Likes: {getattr(info, 'likes', 'N/A')}\n"
                f"Tags: {getattr(info, 'tags', 'N/A')}"
            )
        except Exception as e:
            return f"Failed to fetch info for '{dataset_id}': {e}"


class HFDownloadTool(BaseTool):
    name: str = "hf_dataset_download"
    description: str = "Download a Hugging Face dataset"
    args_schema: type = HFDownloadInput

    def _run(self, dataset_id: str, split: Optional[str] = None, local_dir: str = "./hf_datasets") -> str:
        try:
            # Construct a safe path for saving the dataset
            safe_dataset_id = dataset_id.replace('/', '__')
            path = os.path.join(local_dir, safe_dataset_id)
            
            # Load and save the dataset
            data = load_dataset(dataset_id, split=split) if split else load_dataset(dataset_id)
            data.save_to_disk(path)
            
            return f"Dataset '{dataset_id}' saved to {path}"
        except Exception as e:
            return f"Download failed for '{dataset_id}': {e}"


class HFPreviewTool(BaseTool):
    name: str = "hf_dataset_preview"
    description: str = "Preview rows from a Hugging Face dataset"
    args_schema: type = HFPreviewInput

    def _run(self, dataset_id: str, split: str = "train", num_rows: int = 5) -> str:
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
            preview_data = []
            for i, row in enumerate(ds):
                if i >= num_rows:
                    break
                preview_data.append(row)

            if not preview_data:
                return f"No data found for split '{split}' in dataset '{dataset_id}'."

            # Convert list of dicts to a string representation (similar to DataFrame.to_string)
            header = list(preview_data[0].keys())
            rows = [list(row.values()) for row in preview_data]
            
            # Basic string formatting for the table
            header_str = " | ".join(map(str, header))
            separator = "-|- ".join([ "-" * len(str(h)) for h in header])
            rows_str = "\n".join([" | ".join(map(str, r)) for r in rows])

            return f"{header_str}\n{separator}\n{rows_str}"
            
        except Exception as e:
            return f"Failed to preview '{dataset_id}': {e}"


# ------------------- Toolkit Class -------------------

class HuggingFaceToolkit:
    def __init__(self):
        self.tools = [
            HFSearchTool(),
            HFInfoTool(),
            HFDownloadTool(),
            HFPreviewTool()
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools


