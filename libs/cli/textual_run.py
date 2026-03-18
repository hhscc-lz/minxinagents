from textual_serve.server import Server

server = Server("python -m deepagents_cli.textual_web", title="民心智能体")
server.serve()
