# FILE: api_toolkit.py

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ApiRequestInput(BaseModel):
    """Input schema for the API request tool."""

    url: str = Field(..., description="The full URL of the API endpoint to call.")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = Field(
        "GET", description="The HTTP method to use for the request."
    )
    headers: Optional[Dict[str, str]] = Field(
        None,
        description="A dictionary of HTTP headers to send with the request, e.g., {'Authorization': 'Bearer YOUR_TOKEN'}.",
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="A dictionary of query parameters to append to the URL (for GET requests).",
    )
    json_payload: Optional[Dict[str, Any]] = Field(
        None,
        description="A JSON-serializable dictionary to send as the request body (for POST, PUT, PATCH requests).",
    )
    timeout: int = Field(
        20, description="The timeout in seconds for the request."
    )


class ApiRequestTool(BaseTool):
    """A tool for making generic HTTP requests to any REST API."""

    name: str = "api_request"
    description: str = (
        "🔌 Makes a generic HTTP request to any REST API endpoint. Use this for services that don't have a dedicated toolkit.\n\n"
        "✅ Use Cases:\n"
        "  - Fetching data from public APIs (e.g., weather, public data sources).\n"
        "  - Interacting with internal company microservices.\n"
        "  - Performing actions on platforms with a REST API (e.g., creating a ticket, fetching user data).\n\n"
        "🔑 Authentication:\n"
        "  - For API keys or bearer tokens, use the `headers` parameter.\n"
        "  - Example: `{'Authorization': 'Bearer YOUR_SECRET_TOKEN', 'X-Api-Key': 'YOUR_API_KEY'}`\n\n"
        "⚙️ Method-Specifics:\n"
        "  - **GET**: Use `params` to specify URL query parameters.\n"
        "  - **POST/PUT/PATCH**: Use `json_payload` to specify the request body.\n\n"
        "🚫 Avoid:\n"
        "  - Trying to download files (use the `web_toolkit` for that).\n"
        "  - Making requests to local or private network addresses unless explicitly allowed."
    )
    args_schema: Type[BaseModel] = ApiRequestInput

    def _run(
        self,
        url: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        timeout: int = 20,
    ) -> str:
        """Executes the API request and returns the response."""
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_payload,
                timeout=timeout,
            )

            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Try to parse and return JSON, fall back to text
            try:
                parsed_json = response.json()
                # Truncate large responses to avoid overwhelming the agent context
                response_str = json.dumps(parsed_json, indent=2)
                if len(response_str) > 3000:
                    response_str = response_str[:3000] + "\n\n... (response truncated)"
                return f"✅ Success (Status {response.status_code}):\n{response_str}"
            except json.JSONDecodeError:
                # Handle non-JSON responses
                content = response.text
                if not content:
                    return f"✅ Success (Status {response.status_code}): No content returned."
                if len(content) > 3000:
                    content = content[:3000] + "\n\n... (response truncated)"
                return f"✅ Success (Status {response.status_code}):\n{content}"

        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP errors specifically, providing more context
            error_content = http_err.response.text
            try:
                # Try to parse error content as JSON for better readability
                error_json = json.loads(error_content)
                error_details = json.dumps(error_json, indent=2)
            except json.JSONDecodeError:
                error_details = error_content

            return (
                f"❌ HTTP Error (Status {http_err.response.status_code}): {http_err.response.reason}\n"
                f"URL: {http_err.request.url}\n"
                f"Response Body: {error_details}"
            )
        except requests.exceptions.RequestException as e:
            # Handle other request exceptions (e.g., connection error, timeout)
            return f"❌ Request Failed: {type(e).__name__} - {str(e)}"
        except Exception as e:
            # Catch any other unexpected errors
            return f"❌ An unexpected error occurred: {str(e)}"


class ApiToolkit:
    """Toolkit for interacting with generic REST APIs."""

    def __init__(self):
        self.tools = [ApiRequestTool()]

    def get_tools(self) -> List[BaseTool]:
        return self.tools


# Example usage for testing
if __name__ == "__main__":
    api_toolkit = ApiToolkit()
    request_tool = api_toolkit.get_tools()[0]

    print("--- 1. Testing GET request ---")
    get_result = request_tool.invoke(
        {
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "method": "GET",
        }
    )
    print(get_result)

    print("\n--- 2. Testing POST request ---")
    post_result = request_tool.invoke(
        {
            "url": "https://jsonplaceholder.typicode.com/posts",
            "method": "POST",
            "json_payload": {
                "title": "foo",
                "body": "bar",
                "userId": 1,
            },
            "headers": {"Content-type": "application/json; charset=UTF-8"},
        }
    )
    print(post_result)

    print("\n--- 3. Testing HTTP Error ---")
    error_result = request_tool.invoke(
        {"url": "https://jsonplaceholder.typicode.com/posts/99999", "method": "GET"}
    )
    print(error_result)


