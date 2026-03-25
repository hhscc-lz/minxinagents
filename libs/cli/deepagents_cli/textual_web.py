"""Web-friendly entrypoint for serving the Textual CLI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from deepagents_cli.main import run_textual_cli_async


def main() -> int:
    """Run the Textual CLI through the standard interactive launcher.

    assistant_id 从环境变量 MINXIN_USER 读取（由 textual_run.py 的认证层注入）。
    未设置时回落到 "agent"，仅用于本地开发调试。

    每个用户使用独立的工作目录 ~/.deepagents/{assistant_id}/workspace/，
    避免不同用户生成的文件互相可见。
    """
    assistant_id = os.environ.get("MINXIN_USER", "shared-session")

    # 切换到用户专属工作目录，Agent 写出的文件只在该目录下
    workspace = Path.home() / ".deepagents" / assistant_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)

    result = asyncio.run(run_textual_cli_async(assistant_id))
    return result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
