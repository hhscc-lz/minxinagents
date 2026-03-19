"""Custom tools for the CLI agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tavily import TavilyClient

_UNSET = object()
_tavily_client: TavilyClient | object | None = _UNSET


def _get_tavily_client() -> TavilyClient | None:
    """Get or initialize the lazy Tavily client singleton.

    Returns:
        TavilyClient instance, or None if API key is not configured.
    """
    global _tavily_client  # noqa: PLW0603  # Module-level cache requires global statement
    if _tavily_client is not _UNSET:
        return _tavily_client  # type: ignore[return-value]  # narrowed by sentinel check

    from deepagents_cli.config import settings

    if settings.has_tavily:
        from tavily import TavilyClient as _TavilyClient

        _tavily_client = _TavilyClient(api_key=settings.tavily_api_key)
    else:
        _tavily_client = None
    return _tavily_client


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    import requests

    try:
        kwargs: dict[str, Any] = {}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(method.upper(), url, timeout=timeout, **kwargs)

        try:
            content = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            content = response.text

        return {
            "success": response.status_code < 400,  # noqa: PLR2004  # HTTP status code threshold
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }


def web_search(  # noqa: ANN201  # Return type depends on dynamic tool configuration
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    try:
        import requests
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install 'deepagents[cli]'",
            "query": query,
        }

    client = _get_tavily_client()
    if client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        import requests
        from markdownify import markdownify
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install 'deepagents[cli]'",
            "url": url,
        }

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}


def export_data(sql: str) -> dict[str, Any]:
    """Execute a SQL query and export the result set to object storage.

    This tool runs the given SELECT statement, writes the results to an Excel file,
    and uploads it to the configured object storage (MinIO).  The caller receives
    a pre-signed download URL valid for 12 hours.

    Use this tool when the user asks to export, download, or save query results
    — for example detail records filtered by region, date range, or status.

    Args:
        sql: The SQL SELECT statement to execute.

    Returns:
        Dictionary containing:
        - success: Whether the export completed successfully
        - url: Pre-signed download URL for the exported file (valid 12 hours)

    IMPORTANT: After using this tool:
    1. The download URL is automatically displayed to the user by the interface — do NOT repeat or output the URL in your response.
    2. Only tell the user the export succeeded and how many rows were exported.
    3. If the export failed, relay the error message to the user.
    """
    import io
    import os
    import uuid
    from datetime import timedelta

    import pandas as pd
    import pymssql
    from minio import Minio

    _MINIO_INTERNAL = "172.17.3.61:8882"
    _MINIO_PUBLIC = "202.97.181.107:8882"
    _BUCKET = "minxinagent"
    _MAX_ROWS = 100_000

    _COLUMN_LABELS: dict[str, str] = {
        "oriid": "诉求编号",
        "tousu_id": "部门编号",
        "tsly": "投诉来源",
        "by_area": "城市",
        "by_qx": "区县",
        "ai_xiaoqu": "小区",
        "mpeach_date": "诉求时间",
        "mpeach_text": "诉求标题",
        "mpeach_gut": "诉求内容",
        "mpeach_dx": "一级定性",
        "mpeach_wtlb": "二级定性",
        "mpeach_wtxl": "三级定性",
        "aj_blzt": "案件办理状态",
        "sqxz": "诉求性质",
        "blfs": "办理方式",
        "jjqk": "部门解决情况",
        "bl_bmmc": "办理部门",
        "bj_date": "办结时间",
        "fenpai_date": "分派时间",
        "niban_date": "拟办时间",
        "pingxing_date": "诉求人评价时间",
        "manyidu": "满意度",
        "tui_num": "退件次数",
    }

    try:
        # 1. Execute SQL and load into DataFrame
        conn = pymssql.connect(
            server=os.environ["DB_HOST"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            database=os.environ["DB_NAME"],
        )
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()

        # 2. Row limit check
        if len(df) > _MAX_ROWS:
            return {
                "success": False,
                "url": "",
                "error": f"数据量 {len(df):,} 条超过导出上限（{_MAX_ROWS:,} 条），请缩小查询范围后重试。",
            }

        # 3. Keep only known columns and rename to Chinese labels
        known_cols = [c for c in df.columns if c in _COLUMN_LABELS]
        df = df[known_cols].rename(columns=_COLUMN_LABELS)

        # 4. Write to Excel in memory
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        file_size = buffer.getbuffer().nbytes

        # 5. Upload to MinIO via internal address
        object_name = (
            f"exports/{pd.Timestamp.now().strftime('%Y%m%d')}/"
            f"{uuid.uuid4().hex[:12]}.xlsx"
        )
        client = Minio(
            _MINIO_INTERNAL,
            access_key=os.environ["MINIO_ROOT_USER"],
            secret_key=os.environ["MINIO_ROOT_PASSWORD"],
            secure=False,
        )
        client.put_object(
            _BUCKET,
            object_name,
            buffer,
            length=file_size,
            content_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )

        # 6. Generate presigned URL using public-facing client so the signature
        #    is computed against the public host (replacing the host after signing
        #    would invalidate the signature).
        public_client = Minio(
            _MINIO_PUBLIC,
            access_key=os.environ["MINIO_ROOT_USER"],
            secret_key=os.environ["MINIO_ROOT_PASSWORD"],
            secure=False,
        )
        public_url = public_client.presigned_get_object(
            _BUCKET,
            object_name,
            expires=timedelta(hours=12),
        )

        return {"success": True, "url": public_url}

    except Exception as e:  # noqa: BLE001
        return {"success": False, "url": "", "error": str(e)}
