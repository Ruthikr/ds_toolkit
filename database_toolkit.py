from __future__ import annotations
import os
import re
import logging
import json
import tempfile
from typing import Any, List, Type, Optional, Union
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages a single, global database connection with connection pooling."""
    _engine: Engine | None = None
    _connection_string: str = ""
    _dialect: str = ""

    @classmethod
    def get_engine(cls) -> Engine:
        if cls._engine is None:
            raise ConnectionError("Database connection not established. Use 'db_connect' first.")
        return cls._engine

    @classmethod
    def get_connection_string(cls) -> str:
        return cls._connection_string

    @classmethod
    def get_dialect(cls) -> str:
        return cls._dialect

    @classmethod
    def connect(cls, connection_string: str) -> None:
        original_string = connection_string
        
        # Handle simple file paths
        if not re.match(r"^\w+://", connection_string):
            # Check for file existence with various extensions
            possible_paths = [
                connection_string,
                f"{connection_string}.db",
                f"{connection_string}.sqlite",
                f"./{connection_string}",
                f"./{connection_string}.db",
                f"./{connection_string}.sqlite"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    connection_string = f"sqlite:///{abs_path}"
                    logger.info(f"Auto-formatted SQLite path: {connection_string}")
                    break
            else:
                raise ValueError(f"No database file found at: {original_string}")
        
        # Create engine
        try:
            cls._engine = create_engine(connection_string, pool_pre_ping=True)
            cls._connection_string = connection_string
            cls._dialect = cls._engine.dialect.name
            
            # Test connection
            with cls._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Connected to {cls._dialect} database: {cls._mask_credentials(connection_string)}")
        except Exception as e:
            if cls._engine:
                cls._engine.dispose()
                cls._engine = None
            raise ConnectionError(f"Connection failed: {str(e)}")

    @classmethod
    def disconnect(cls) -> None:
        if cls._engine:
            cls._engine.dispose()
            logger.info(f"Disconnected from {cls._dialect} database: {cls._mask_credentials(cls._connection_string)}")
            cls._engine = None
            cls._connection_string = ""
            cls._dialect = ""

    @staticmethod
    def _mask_credentials(connection_string: str) -> str:
        """Mask sensitive credentials in connection strings"""
        if "://" not in connection_string:
            return connection_string
            
        # Mask password in connection string
        protocol, rest = connection_string.split("://", 1)
        if "@" in rest:
            user_part, host_part = rest.split("@", 1)
            if ":" in user_part:
                user, password = user_part.split(":", 1)
                return f"{protocol}://{user}:******@{host_part}"
        return f"{protocol}://******"

class BaseDatabaseTool(BaseTool):
    """Base class with common input parsing functionality"""
    def parse_input(self, input_data: Any) -> dict:
        """Parse flexible input formats into parameter dictionary"""
        params = {}
        
        # Handle empty input
        if input_data is None:
            return {}
            
        # Handle JSON string input
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                if isinstance(data, dict):
                    return data
                else:
                    logger.warning("JSON string didn't parse to dict")
            except json.JSONDecodeError:
                # Not JSON, continue to other formats
                pass
        
        # Handle dictionary input
        if isinstance(input_data, dict):
            return input_data
        
        # Handle string input for all tools
        if isinstance(input_data, str):
            # Handle tools that accept single string parameters
            tool_mappings = {
                "db_connect": ["connection_string"],
                "db_list_tables": ["schema_name"],
                "db_disconnect": [],
                "db_get_schema": ["table_name"],
                "db_preview_table": ["table_name"],
                "db_get_row_count": ["table_name"],
                "db_query": ["query"],
                "db_download_csv": ["query"]
            }
            
            if self.name in tool_mappings:
                param_names = tool_mappings[self.name]
                if param_names:
                    # For tools with a single parameter
                    return {param_names[0]: input_data}
                else:
                    # For tools that don't need parameters
                    return {}
        
        return {"input": input_data}

class ConnectInput(BaseModel):
    connection_string: str = Field(
        ...,
        description="Database connection string. Examples: "
        "'sample.db' (SQLite file in current directory), "
        "'sqlite:///path/to/database.db', "
        "'postgresql://user:pass@localhost/dbname', "
        "'mysql+pymysql://user:pass@localhost/mydb'"
    )

class ConnectTool(BaseDatabaseTool):
    name: str = "db_connect"
    description: str = (
        "ESTABLISHES DATABASE CONNECTION. MUST BE CALLED FIRST! "
        "For SQLite: 'sample.db' or 'sqlite:///path/to/db'. "
        "For others: 'protocol://user:pass@host/dbname'. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'sample.db'"
        "\n- JSON: {'connection_string': 'sample.db'}"
    )
    args_schema: Type[BaseModel] = ConnectInput

    def _run(self, input_data: Any) -> str:
        try:
            params = self.parse_input(input_data)
            connection_string = params.get("connection_string", "")
            
            if not connection_string:
                return "❌ Missing connection_string parameter"
                
            DatabaseConnection.connect(connection_string)
            return "✅ Connection successful. You can now use other database tools."
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return (
                f"❌ Connection failed: {str(e)}\n"
                "CONNECTION EXAMPLES:\n"
                "1. SQLite: 'sample.db' (file in current dir)\n"
                "2. SQLite: 'sqlite:///path/to/sample.db'\n"
                "3. PostgreSQL: 'postgresql://user:pass@localhost:5432/mydb'\n"
                "4. MySQL: 'mysql+pymysql://user:pass@localhost/mydb'"
            )

class DisconnectTool(BaseDatabaseTool):
    name: str = "db_disconnect"
    description: str = (
        "CLOSES THE DATABASE CONNECTION. Always call this when finished to release resources. "
        "No input parameters needed."
    )
    args_schema: Type[BaseModel] = type('EmptySchema', (BaseModel,), {})

    def _run(self, input_data: Any = None) -> str:
        try:
            DatabaseConnection.disconnect()
            return "✅ Database connection closed."
        except Exception as e:
            logger.error(f"Disconnection failed: {str(e)}")
            return f"❌ Disconnection failed: {str(e)}"

class ListTablesInput(BaseModel):
    schema_name: Optional[str] = Field(
        None, 
        description="Optional schema name (e.g., 'public' in PostgreSQL)"
    )

class ListTablesTool(BaseDatabaseTool):
    name: str = "db_list_tables"
    description: str = (
        "LISTS ALL TABLES in the connected database. "
        "For databases with schemas, optionally specify schema_name. "
        "\n\nINPUT FORMATS:"
        "\n- Empty: ''"
        "\n- String: 'public' (schema name)"
        "\n- JSON: {'schema_name': 'public'}"
    )
    args_schema: Type[BaseModel] = ListTablesInput

    def _run(self, input_data: Any = None) -> str:
        try:
            params = self.parse_input(input_data)
            schema_name = params.get("schema_name", None)
            
            engine = DatabaseConnection.get_engine()
            dialect = DatabaseConnection.get_dialect()
            
            # Handle SQLite specially
            if dialect == "sqlite":
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"))
                    tables = [row[0] for row in result.fetchall()]
            else:
                inspector = inspect(engine)
                tables = inspector.get_table_names(schema=schema_name)
                
            return f"📋 Tables: {', '.join(tables)}"
        except Exception as e:
            logger.error(f"Failed to list tables: {str(e)}")
            return (
                f"❌ Failed to list tables: {str(e)}\n"
                "TROUBLESHOOTING:"
                "\n1. Did you call db_connect first?"
                "\n2. For schemas, use db_list_tables with schema name"
            )

class GetSchemaInput(BaseModel):
    table_name: str = Field(..., description="Name of the table to inspect")
    schema_name: Optional[str] = Field(
        None, 
        description="Optional schema name (e.g., 'public' in PostgreSQL)"
    )

class GetSchemaTool(BaseDatabaseTool):
    name: str = "db_get_schema"
    description: str = (
        "RETRIEVES TABLE SCHEMA including columns, data types, keys, and indexes. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'employees' (table name)"
        "\n- String: 'public.employees' (schema.table)"
        "\n- JSON: {'table_name': 'employees', 'schema_name': 'public'}"
    )
    args_schema: Type[BaseModel] = GetSchemaInput

    def _run(self, input_data: Any) -> str:
        try:
            params = self.parse_input(input_data)
            table_name = params.get("table_name", "")
            schema_name = params.get("schema_name", None)
            
            # Handle "schema.table" format in string input
            if '.' in table_name:
                parts = table_name.split('.')
                if len(parts) == 2:
                    schema_name, table_name = parts
            
            if not table_name:
                return "❌ Missing table_name parameter"
            
            engine = DatabaseConnection.get_engine()
            inspector = inspect(engine)
            
            # Get columns
            columns = inspector.get_columns(table_name, schema=schema_name)
            column_details = [
                f"{col['name']} ({str(col['type'])})" + 
                (" PRIMARY KEY" if col.get('primary_key', False) else "") +
                (" NULL" if col['nullable'] else " NOT NULL") +
                (f" DEFAULT {col['default']}" if col.get('default') else "")
                for col in columns
            ]
            
            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name)
            pk_info = f"🔑 Primary Keys: {', '.join(pk_constraint['constrained_columns'])}" if pk_constraint and pk_constraint['constrained_columns'] else ""
            
            # Get foreign keys
            fk_constraints = inspector.get_foreign_keys(table_name, schema=schema_name)
            fk_info = ""
            if fk_constraints:
                fk_details = [
                    f"{', '.join(fk['constrained_columns'])} → {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                    for fk in fk_constraints
                ]
                fk_info = f"\n🔗 Foreign Keys:\n\t- " + "\n\t- ".join(fk_details)
            
            # Get indexes
            indexes = inspector.get_indexes(table_name, schema=schema_name)
            idx_info = ""
            if indexes:
                idx_details = [
                    f"{idx['name']}: {', '.join(idx['column_names'])}"
                    for idx in indexes
                ]
                idx_info = f"\n🔍 Indexes:\n\t- " + "\n\t- ".join(idx_details)
            
            return (
                f"📊 Schema for '{table_name}':\n"
                f"• Columns:\n\t- " + "\n\t- ".join(column_details) + 
                (f"\n• {pk_info}" if pk_info else "") +
                (fk_info if fk_info else "") +
                (idx_info if idx_info else "")
            )
        except Exception as e:
            logger.error(f"Schema retrieval failed: {str(e)}")
            return (
                f"❌ Failed to get schema: {str(e)}\n"
                "TROUBLESHOOTING:"
                "\n1. Verify table exists with db_list_tables"
                "\n2. Check schema requirements (use 'schema.table' format)"
                "\n3. Ensure connection is active"
            )

class QueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    limit_rows: Optional[int] = Field(
        10, 
        description="Max rows to return (default: 100, -1 for all)"
    )

class QueryTool(BaseDatabaseTool):
    name: str = "db_query"
    description: str = (
        "EXECUTES SQL QUERIES (SELECT/INSERT/UPDATE/DELETE). "
        "SELECT: returns first 100 rows by default. "
        "Others: returns affected row count. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'SELECT * FROM employees'"
        "\n- JSON: {'query': 'UPDATE...', 'limit_rows': -1}"
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input_data: Any) -> str:
        try:
            params = self.parse_input(input_data)
            query = params.get("query", "")
            limit_rows = params.get("limit_rows", 10)
            
            if not query:
                return "❌ Missing query parameter"
            
            # Clean and sanitize query
            query = query.strip()
            if query.endswith(';'):
                query = query[:-1]
            
            engine = DatabaseConnection.get_engine()
            
            # Handle multiple statements
            if ';' in query and not query.strip().lower().startswith('create'):
                return "❌ Only single-statement queries are supported"
            
            # Determine query type
            is_select = re.match(r'^\s*select\b', query, re.IGNORECASE) is not None
            
            with engine.begin() as conn:  # Use transaction
                # Handle SELECT queries
                if is_select:
                    # Apply LIMIT if not already present and limit is specified
                    modified_query = query
                    if limit_rows != -1 and "limit" not in query.lower():
                        # Only add LIMIT if not already in query
                        modified_query += f" LIMIT {limit_rows}"
                    
                    df = pd.read_sql(modified_query, conn)
                    
                    if df.empty:
                        return "✅ Query executed successfully. No results returned."
                        
                    return (
                        f"🔍 Query Results ({len(df)} rows):\n"
                        f"{df.to_string(index=False)}\n"
                        f"ℹ️ Use db_download_csv for full dataset"
                    )
                
                # Handle other query types (INSERT/UPDATE/DELETE)
                result = conn.execute(text(query))
                rowcount = result.rowcount
                return f"✅ Query executed successfully. Affected rows: {rowcount}"
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return (
                f"❌ Query failed: {str(e)}\n"
                "TROUBLESHOOTING:"
                "\n1. Check SQL syntax"
                "\n2. Verify table names with db_list_tables"
                "\n3. Ensure query is compatible with database type"
            )

class DownloadInput(BaseModel):
    query: str = Field(..., description="SQL SELECT query")
    file_path: Optional[str] = Field(
        None,
        description="Optional output file path (default: system temp file)"
    )

class DownloadTool(BaseDatabaseTool):
    name: str = "db_download_csv"
    description: str = (
        "EXPORTS QUERY RESULTS TO CSV FILE. Only works with SELECT queries. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'SELECT * FROM employees' (file_path will be auto-generated)"
        "\n- JSON: {'query': 'SELECT...', 'file_path': 'output.csv'}"
    )
    args_schema: Type[BaseModel] = DownloadInput

    def _run(self, input_data: Any) -> str:
        try:
            # Use base class parsing for consistent input handling
            params = self.parse_input(input_data)
            query = params.get("query", "")
            file_path = params.get("file_path", None)
            
            if not query:
                return "❌ Missing query parameter"
            
            # Clean and validate query
            query = query.strip()
            if query.endswith(';'):
                query = query[:-1]
                
            # Robust SQL validation that handles different formatting
            normalized_query = query.lower().strip()
            is_select = (
                normalized_query.startswith("select") or
                normalized_query.startswith("select") or
                normalized_query.startswith("(select") or
                normalized_query.startswith("select\n") or
                normalized_query.startswith("select\r") or
                normalized_query.startswith("select\t") or
                normalized_query.startswith("select /*")
            )
                
            if not is_select:
                return "❌ Only SELECT queries can be exported to CSV"
                
            # Generate file path if not provided
            if not file_path:
                file_path = f"{tempfile.gettempdir()}/export_{os.getpid()}.csv"
                logger.info(f"Generated temporary file path: {file_path}")
                
            # Create directory if needed
            directory = os.path.dirname(file_path) or "."
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")

            # Execute and save
            engine = DatabaseConnection.get_engine()
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, engine)
            
            logger.info(f"Saving {len(df)} rows to {file_path}")
            df.to_csv(file_path, index=False)
            
            return f"💾 Saved {len(df)} rows to {os.path.abspath(file_path)}"
        except Exception as e:
            logger.error(f"Export failed: {str(e)}", exc_info=True)
            return (
                f"❌ Export failed: {str(e)}\n"
                "TROUBLESHOOTING:"
                "\n1. Verify query starts with SELECT (case insensitive)"
                "\n2. Check table names with db_list_tables"
                "\n3. Ensure proper input format:"
                "\n   - JSON: {'query': 'SELECT...', 'file_path': 'output.csv'}"
                "\n   - String: 'SELECT * FROM table'"
            )

class PreviewTableInput(BaseModel):
    table_name: str = Field(..., description="Table to preview")
    limit: Optional[int] = Field(5, description="Rows to show (default: 5)")
    schema_name: Optional[str] = Field(
        None, 
        description="Optional schema name"
    )

class PreviewTableTool(BaseDatabaseTool):
    name: str = "db_preview_table"
    description: str = (
        "PREVIEWS TABLE CONTENTS. Shows first 5 rows by default. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'employees'"
        "\n- String: 'public.employees'"
        "\n- JSON: {'table_name': 'employees', 'limit': 10}"
    )
    args_schema: Type[BaseModel] = PreviewTableInput

    def _run(self, input_data: Any) -> str:
        try:
            params = self.parse_input(input_data)
            table_name = params.get("table_name", "")
            schema_name = params.get("schema_name", None)
            limit = params.get("limit", 5)
            
            # Handle "schema.table" format in string input
            if '.' in table_name:
                parts = table_name.split('.')
                if len(parts) == 2:
                    schema_name, table_name = parts
            
            if not table_name:
                return "❌ Missing table_name parameter"
            
            engine = DatabaseConnection.get_engine()
            dialect = DatabaseConnection.get_dialect()
            
            # Handle schema for table name
            full_table_name = table_name
            if schema_name and dialect != "sqlite":
                full_table_name = f"{schema_name}.{table_name}"
                
            df = pd.read_sql(f"SELECT * FROM {full_table_name} LIMIT {limit}", engine)
            
            if df.empty:
                return f"ℹ️ Table '{table_name}' is empty"
                
            return (
                f"👀 Preview of '{table_name}' ({len(df)} rows):\n"
                f"{df.to_string(index=False)}"
            )
        except Exception as e:
            logger.error(f"Preview failed: {str(e)}")
            return (
                f"❌ Preview failed: {str(e)}\n"
                "TROUBLESHOOTING:"
                "\n1. Verify table exists with db_list_tables"
                "\n2. Check schema requirements (use 'schema.table' format)"
                "\n3. Ensure connection is active"
            )

class RowCountInput(BaseModel):
    table_name: str = Field(..., description="Table to count rows in")
    where_clause: Optional[str] = Field(
        None, 
        description="Optional filter condition (without WHERE keyword)"
    )
    schema_name: Optional[str] = Field(
        None, 
        description="Optional schema name"
    )

class RowCountTool(BaseDatabaseTool):
    name: str = "db_get_row_count"
    description: str = (
        "COUNTS ROWS IN A TABLE. Optionally filter with a WHERE clause. "
        "\n\nINPUT FORMATS:"
        "\n- String: 'employees'"
        "\n- String: 'employees WHERE salary > 50000'"
        "\n- JSON: {'table_name': 'employees', 'where_clause': 'department = \"Engineering\"'}"
    )
    args_schema: Type[BaseModel] = RowCountInput

    def _run(self, input_data: Any) -> str:
        try:
            params = self.parse_input(input_data)
            table_name = params.get("table_name", "")
            where_clause = params.get("where_clause", None)
            schema_name = params.get("schema_name", None)
            
            # Handle "schema.table" format in string input
            if '.' in table_name:
                parts = table_name.split('.')
                if len(parts) == 2:
                    schema_name, table_name = parts
            
            # Handle "table WHERE condition" format
            if not where_clause and ' where ' in table_name.lower():
                parts = re.split(r'\s+where\s+', table_name, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) == 2:
                    table_name = parts[0].strip()
                    where_clause = parts[1].strip()
            
            if not table_name:
                return "❌ Missing table_name parameter"
            
            engine = DatabaseConnection.get_engine()
            dialect = DatabaseConnection.get_dialect()
            
            # Handle schema for table name
            full_table_name = table_name
            if schema_name and dialect != "sqlite":
                full_table_name = f"{schema_name}.{table_name}"
                
            # Build query
            base_query = f"SELECT COUNT(*) AS row_count FROM {full_table_name}"
            if where_clause:
                base_query += f" WHERE {where_clause.strip()}"
                
            result = pd.read_sql(base_query, engine)
            count = result.iloc[0]['row_count']
            
            return f"🧮 Row count for '{table_name}': {count}"
        except Exception as e:
            logger.error(f"Row count failed: {str(e)}")
            return (
                f"❌ Row count failed: {str(e)}\n"
                "INPUT EXAMPLES:"
                "\n1. 'employees'"
                "\n2. 'employees WHERE department = \"Sales\"'"
                "\n3. {'table_name': 'orders', 'where_clause': 'status = \"shipped\"'}"
            )

class DatabaseToolkit:
    """Comprehensive toolkit for SQL database interaction designed for AI agents"""
    def __init__(self):
        self.tools = [
            ConnectTool(),
            DisconnectTool(),
            ListTablesTool(),
            GetSchemaTool(),
            PreviewTableTool(),
            RowCountTool(),
            QueryTool(),
            DownloadTool(),
        ]

    def get_tools(self) -> List[BaseTool]:
        return self.tools

    def get_connection_status(self) -> str:
        """Returns current connection status for agent awareness"""
        try:
            conn_str = DatabaseConnection.get_connection_string()
            return f"✅ Connected to {DatabaseConnection.get_dialect()}: {DatabaseConnection._mask_credentials(conn_str)}"
        except ConnectionError:
            return "❌ Not connected to any database"