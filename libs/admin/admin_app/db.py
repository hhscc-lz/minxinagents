"""SQLite reading logic for the admin dashboard.

Self-contained — no deepagents_cli imports.
Ported from libs/cli/deepagents_cli/sessions.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_aiosqlite_patched = False
_SQLITE_MAX_VARIABLE_NUMBER = 500


def get_db_path() -> Path:
    return Path.home() / ".deepagents" / "sessions.db"


def _patch_aiosqlite() -> None:
    """Patch aiosqlite.Connection with `is_alive()` if missing.
    Required by langgraph-checkpoint>=2.1.0.
    """
    global _aiosqlite_patched  # noqa: PLW0603
    if _aiosqlite_patched:
        return
    import aiosqlite as _aiosqlite

    if not hasattr(_aiosqlite.Connection, "is_alive"):
        def _is_alive(self: _aiosqlite.Connection) -> bool:
            return bool(self._running and self._connection is not None)
        _aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]

    _aiosqlite_patched = True


@asynccontextmanager
async def _connect():
    import aiosqlite as _aiosqlite
    _patch_aiosqlite()
    async with _aiosqlite.connect(str(get_db_path()), timeout=30.0) as conn:
        yield conn


async def _table_exists(conn, table: str) -> bool:
    async with conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ) as cursor:
        return await cursor.fetchone() is not None


def _get_serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

async def count_stats() -> dict[str, int]:
    """Return total thread count and today's new thread count."""
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return {"total_threads": 0, "today_threads": 0}

        async with conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
        ) as cursor:
            row = await cursor.fetchone()
            total = row[0] if row else 0

        today = datetime.now(tz=timezone.utc).date().isoformat()
        async with conn.execute(
            """
            SELECT COUNT(DISTINCT thread_id) FROM checkpoints
            WHERE json_extract(metadata, '$.updated_at') >= ?
            """,
            (today,),
        ) as cursor:
            row = await cursor.fetchone()
            today_count = row[0] if row else 0

    return {"total_threads": total, "today_threads": today_count}


# ---------------------------------------------------------------------------
# Thread list
# ---------------------------------------------------------------------------

async def list_threads(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List threads ordered by last activity, with message_count and initial_prompt."""
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        async with conn.execute(
            """
            SELECT thread_id,
                   json_extract(metadata, '$.agent_name') as agent_name,
                   MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                   MIN(json_extract(metadata, '$.updated_at')) as created_at,
                   MAX(checkpoint_id) as latest_checkpoint_id
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        threads = [
            {
                "thread_id": r[0],
                "agent_name": r[1],
                "updated_at": r[2],
                "created_at": r[3],
                "latest_checkpoint_id": r[4],
                "message_count": 0,
                "initial_prompt": None,
            }
            for r in rows
        ]

        if threads:
            serde = _get_serde()
            thread_ids = [t["thread_id"] for t in threads]
            summaries = await _load_summaries_batch(conn, thread_ids, serde)
            for t in threads:
                s = summaries.get(t["thread_id"])
                if s:
                    t["message_count"] = s["message_count"]
                    t["initial_prompt"] = s["initial_prompt"]

    return threads


async def _load_summaries_batch(
    conn, thread_ids: list[str], serde
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    loop = asyncio.get_running_loop()

    for start in range(0, len(thread_ids), _SQLITE_MAX_VARIABLE_NUMBER):
        chunk = thread_ids[start: start + _SQLITE_MAX_VARIABLE_NUMBER]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT thread_id, type, checkpoint FROM (
                SELECT thread_id, type, checkpoint,
                       ROW_NUMBER() OVER (
                           PARTITION BY thread_id ORDER BY checkpoint_id DESC
                       ) AS rn
                FROM checkpoints
                WHERE thread_id IN ({placeholders})
            ) WHERE rn = 1
        """
        async with conn.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()

        for tid, type_str, blob in rows:
            if not type_str or not blob:
                results[tid] = {"message_count": 0, "initial_prompt": None}
                continue
            try:
                data = await loop.run_in_executor(
                    None, serde.loads_typed, (type_str, blob)
                )
                results[tid] = _summarize_checkpoint(data)
            except Exception:
                logger.warning("Failed to deserialize checkpoint for %s", tid, exc_info=True)
                results[tid] = {"message_count": 0, "initial_prompt": None}

    return results


def _summarize_checkpoint(data: object) -> dict[str, Any]:
    messages = _checkpoint_messages(data)
    return {
        "message_count": len(messages),
        "initial_prompt": _initial_prompt_from_messages(messages),
    }


def _checkpoint_messages(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    channel_values = data.get("channel_values")
    if not isinstance(channel_values, dict):
        return []
    messages = channel_values.get("messages")
    if not isinstance(messages, list):
        return []
    return messages


def _initial_prompt_from_messages(messages: list[object]) -> str | None:
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            return _coerce_prompt_text(getattr(msg, "content", None))
    return None


def _coerce_prompt_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append(str(part))
        joined = " ".join(parts).strip()
        return joined or None
    if content is None:
        return None
    return str(content)


# ---------------------------------------------------------------------------
# Thread messages
# ---------------------------------------------------------------------------

async def get_messages(thread_id: str) -> list[dict[str, Any]]:
    """Return all messages for a thread, deserialized and JSON-serializable."""
    _patch_aiosqlite()
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = str(get_db_path())
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        tup = await saver.aget_tuple(config)

    if not tup or not tup.checkpoint:
        return []

    raw_messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
    result = []
    for msg in raw_messages:
        serialized = _serialize_message(msg)
        if serialized:
            result.append(serialized)
    return result


def _serialize_message(msg: Any) -> dict[str, Any] | None:
    msg_type = getattr(msg, "type", None)
    if msg_type not in ("human", "ai", "tool"):
        return None

    content = getattr(msg, "content", "")
    if isinstance(content, list):
        # Extract text parts and tool_use blocks separately
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input"),
                    })
            else:
                text_parts.append(str(block))
        content_str = "\n".join(t for t in text_parts if t)
    else:
        content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        tool_calls = []

    # Also check .tool_calls attribute (OpenAI-style)
    if msg_type == "ai" and not tool_calls:
        attr_tool_calls = getattr(msg, "tool_calls", None)
        if attr_tool_calls:
            tool_calls = [
                {"id": tc.get("id"), "name": tc.get("name"), "input": tc.get("args", tc.get("input"))}
                for tc in attr_tool_calls
                if isinstance(tc, dict)
            ]

    # Skip empty AI messages that are just internal state
    if msg_type == "ai" and not content_str.strip() and not tool_calls:
        return None

    result: dict[str, Any] = {
        "type": msg_type,
        "content": content_str,
        "id": getattr(msg, "id", None),
    }

    if tool_calls:
        result["tool_calls"] = tool_calls

    if msg_type == "tool":
        result["tool_name"] = getattr(msg, "name", None)
        result["tool_call_id"] = getattr(msg, "tool_call_id", None)

    return result


# ---------------------------------------------------------------------------
# Relative time helper (for API response)
# ---------------------------------------------------------------------------

def relative_time(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
    except (ValueError, TypeError):
        return ""
    delta = datetime.now(tz=dt.tzinfo) - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "刚刚"
    if seconds < 60:
        return f"{seconds}秒前"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟前"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}小时前"
    days = hours // 24
    return f"{days}天前"
