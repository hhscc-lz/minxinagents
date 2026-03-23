"""FastAPI admin dashboard for 民心智能体."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

import admin_app.db as db

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db._patch_aiosqlite()
    yield


app = FastAPI(title="民心智能体 管理后台", lifespan=lifespan, root_path="/admin")


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(_STATIC_DIR / "dashboard.html", media_type="text/html")


@app.get("/api/stats")
async def stats():
    return await db.count_stats()


@app.get("/api/threads")
async def threads(limit: int = 50, offset: int = 0):
    items = await db.list_threads(limit=limit, offset=offset)
    stats = await db.count_stats()
    for t in items:
        t["updated_at_relative"] = db.relative_time(t.get("updated_at"))
        t["created_at_relative"] = db.relative_time(t.get("created_at"))
    return {
        "total": stats["total_threads"],
        "today": stats["today_threads"],
        "threads": items,
    }


@app.get("/api/threads/{thread_id}")
async def thread_detail(thread_id: str):
    messages = await db.get_messages(thread_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "messages": messages}
