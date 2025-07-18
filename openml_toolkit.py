from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import openml
import pandas as pd
import os
from typing import ClassVar, Type

# -------------- Tool 1: Search ------------------
class OpenMLSearchInput(BaseModel):
    query: str = Field(..., description="Keyword to search OpenML datasets")
    max_results: int = Field(5, description="Number of results to return")



class OpenMLSearchTool(BaseTool):
    name: str = "openml_search_datasets_by_tag"
    description: str = "Search for datasets on OpenML by a given tag (e.g., 'medical', 'finance', 'vision'). This is more efficient than a broad keyword search."
    args_schema: ClassVar[Type[BaseModel]] = OpenMLSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            # Use the 'tag' parameter for efficient, server-side filtering
            results = openml.datasets.list_datasets(tag=query, output_format="dataframe")
            if results.empty:
                return f"No datasets found with tag: '{query}'."
            
            # Limit the results locally
            limited_results = results.head(max_results)
            
            return "\n".join([
                f"{row['did']} - {row['name']} ({row['NumberOfInstances']} rows)" 
                for _, row in limited_results.iterrows()
            ])
        except Exception as e:
            return f"Search failed: {e}"


# -------------- Tool 2: Download ------------------
class OpenMLDownloadInput(BaseModel):
    dataset_id: int = Field(..., description="The OpenML dataset ID to download")
    save_path: Optional[str] = Field("openml_datasets", description="Where to save the dataset")

class OpenMLDownloadTool(BaseTool):
    name: str = "openml_download_dataset"
    description: str = "Download an OpenML dataset by ID and save locally as CSV"
    args_schema: ClassVar[Type[BaseModel]] = OpenMLDownloadInput
    

    def _run(self, dataset_id: int, save_path: str = "openml_datasets") -> str:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            df, *_ = dataset.get_data()
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"openml_{dataset_id}.csv")
            df.to_csv(file_path, index=False)
            return f"Dataset {dataset.name} saved to {file_path}"
        except Exception as e:
            return f"Download failed: {e}"


# -------------- Tool 3: Preview ------------------
class OpenMLPreviewInput(BaseModel):
    dataset_id: int = Field(..., description="OpenML dataset ID")
    num_rows: int = Field(5, description="Number of rows to preview")

class OpenMLPreviewTool(BaseTool):
    name: str = "openml_preview_dataset"
    description: str = "Preview top rows of an OpenML dataset"
    args_schema: ClassVar[Type[BaseModel]] = OpenMLPreviewInput
  

    def _run(self, dataset_id: int, num_rows: int = 5) -> str:
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            df, *_ = dataset.get_data()
            return df.head(num_rows).to_string(index=False)
        except Exception as e:
            return f"Preview failed: {e}"


# -------------- Toolkit Wrapper ------------------
class OpenMLToolkit:
    def __init__(self):
        self.tools = [
            OpenMLSearchTool(),
            OpenMLDownloadTool(),
            OpenMLPreviewTool()
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools

if __name__ == "__main__":
    toolkit = OpenMLToolkit()
    tools = toolkit.get_tools()

    print(("=" * 15) + " Search Test " + ("=" * 15))
    print(tools[0].invoke({"query": "heart", "max_results": 10}))

    print(("=" * 15) + " Download Test " + ("=" * 15))
    #print(tools[1].invoke({"dataset_id": 37}))  # diabetes dataset

    print(("=" * 15) + " Preview Test " + ("=" * 15))
  #  print(tools[2].invoke({"dataset_id": 37, "num_rows": 5}))