import os

from textual_serve.server import Server

public_url = os.environ.get("TEXTUAL_PUBLIC_URL", "http://localhost:8000")
server = Server(
    "python -m deepagents_cli.textual_web",
    host="0.0.0.0",
    port=8000,
    public_url=public_url,
    title="民心智能体",
)
server.serve()
