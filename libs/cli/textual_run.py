"""Textual web server entry point with SSO token authentication.

认证流程：
    1. 用户从登录系统跳转：https://your-domain/?token=<加密串>
    2. handle_index 验证 token，将手机号存入短期 session，设置 Cookie
    3. JS 连接 WebSocket 时浏览器自动携带 Cookie
    4. handle_websocket 从 Cookie 读取 session，注入 MINXIN_USER 到子进程
    5. textual_web.py 读取 MINXIN_USER 作为 assistant_id

环境变量：
    TOKEN_SECRET_KEY    与登录系统共享的签名密钥（必须）
    TOKEN_MAX_AGE       token 有效期秒数，默认 300
    TEXTUAL_PUBLIC_URL  对外暴露的 URL，用于 WebSocket 地址
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any

import aiohttp_jinja2
from aiohttp import web
from textual_serve.app_service import AppService
from textual_serve.server import Server

from deepagents_cli.auth import TokenError, verify_token

logger = logging.getLogger(__name__)

public_url = os.environ.get("TEXTUAL_PUBLIC_URL", "http://localhost:8000")

# 短期会话存储：session_key -> phone
# session 在 WebSocket 连接时消费（pop），不会无限增长
_sessions: dict[str, str] = {}


class _AuthAppService(AppService):
    """AppService 子类，支持注入额外环境变量到子进程。"""

    def __init__(self, *args: Any, extra_env: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extra_env = extra_env or {}

    def _build_environment(self, width: int = 80, height: int = 24) -> dict[str, str]:
        env = super()._build_environment(width=width, height=height)
        env.update(self._extra_env)
        return env


class AuthServer(Server):
    """textual_serve.Server 子类，实现 SSO token 认证与用户隔离。"""

    async def handle_index(self, request: web.Request) -> web.Response:
        """验证 token，将手机号存入 session Cookie，返回页面。"""
        router = request.app.router
        font_size = _to_int(request.query.get("fontsize", "16"), 16)
        token = request.query.get("token", "")

        # 验证 token，解析手机号，失败直接拒绝
        try:
            phone = verify_token(token)
            logger.info("用户 %s 认证成功", phone)
        except TokenError as e:
            logger.warning("token 验证失败：%s", e)
            return web.Response(
                status=401,
                content_type="text/html",
                text=f"<!DOCTYPE html><html><head><meta charset='UTF-8'>"
                     f"<title>访问受限</title>"
                     f"<style>body{{font-family:sans-serif;display:flex;align-items:center;"
                     f"justify-content:center;height:100vh;margin:0;background:#f0f2f5;}}"
                     f".box{{text-align:center;color:#555;}}"
                     f"h2{{color:#1a3a6b;}}</style></head>"
                     f"<body><div class='box'>"
                     f"<h2>访问受限</h2>"
                     f"<p>请从系统入口进入民心智能体</p>"
                     f"<p style='font-size:12px;color:#999'>{e}</p>"
                     f"</div></body></html>",
            )

        # 生成短期 session key，存入内存
        session_key = secrets.token_urlsafe(16)
        _sessions[session_key] = phone

        def get_url(route: str, **args: Any) -> str:
            path = router[route].url_for(**args)
            return f"{self.public_url}{path}"

        def get_websocket_url(route: str, **args: Any) -> str:
            url = get_url(route, **args)
            prefix = "wss" if self.public_url.startswith("https") else "ws"
            return prefix + ":" + url.split(":", 1)[1]

        context = {
            "font_size": font_size,
            "app_websocket_url": get_websocket_url("websocket"),  # 不带 token，保持原始格式
            "config": {
                "static": {
                    "url": get_url("static", filename="/").rstrip("/") + "/",
                }
            },
            "application": {"name": self.title},
        }

        response = aiohttp_jinja2.render_template("app_index.html", request, context)
        response.set_cookie(
            "MINXIN_SESSION",
            session_key,
            httponly=True,
            samesite="Lax",
            max_age=60,  # 60 秒内必须建立 WebSocket 连接，否则 session 失效
        )
        return response

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """从 Cookie 读取 session，注入 MINXIN_USER 到子进程环境变量。"""
        session_key = request.cookies.get("MINXIN_SESSION", "")
        phone = _sessions.pop(session_key, "shared-session")  # pop：每个 session 只用一次
        logger.info("WebSocket 连接，用户身份：%s", phone)

        extra_env: dict[str, str] = {"MINXIN_USER": phone}

        websocket = web.WebSocketResponse(heartbeat=15)
        width = _to_int(request.query.get("width", "80"), 80)
        height = _to_int(request.query.get("height", "24"), 24)

        app_service: _AuthAppService | None = None
        try:
            await websocket.prepare(request)
            app_service = _AuthAppService(
                self.command,
                write_bytes=websocket.send_bytes,
                write_str=websocket.send_str,
                close=websocket.close,
                download_manager=self.download_manager,
                debug=self.debug,
                extra_env=extra_env,
            )
            await app_service.start(width, height)
            try:
                await self._process_messages(websocket, app_service)
            finally:
                await app_service.stop()
        except asyncio.CancelledError:
            await websocket.close()
        except Exception as exc:
            logger.exception("WebSocket error: %s", exc)
        finally:
            if app_service is not None:
                await app_service.stop()

        return websocket


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


server = AuthServer(
    "python -m deepagents_cli.textual_web",
    host="0.0.0.0",
    port=8000,
    public_url=public_url,
    title="民心智能体",
)
server.serve()
