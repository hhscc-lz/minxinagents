# ruff: noqa: E501  # Long prompt strings in MEMORY_SYSTEM_PROMPT
"""Middleware for loading agent memory/context from AGENTS.md files.

This module implements support for the AGENTS.md specification (https://agents.md/),
loading memory/context from configurable sources and injecting into the system prompt.

## Overview

AGENTS.md files provide project-specific context and instructions to help AI agents
work effectively. Unlike skills (which are on-demand workflows), memory is always
loaded and provides persistent context.

## Usage

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# Security: FilesystemBackend allows reading/writing from the entire filesystem.
# Either ensure the agent is running within a sandbox OR add human-in-the-loop (HIL)
# approval to file operations.
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",
        "./.deepagents/AGENTS.md",
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## Memory Sources

Sources are simply paths to AGENTS.md files that are loaded in order and combined.
Multiple sources are concatenated in order, with all content included.
Later sources appear after earlier ones in the combined prompt.

## File Format

AGENTS.md files are standard Markdown with no required structure.
Common sections include:
- Project overview
- Build/test commands
- Code style guidelines
- Architecture notes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """State schema for `MemoryMiddleware`.

    Attributes:
        memory_contents: Dict mapping source paths to their loaded content.
            Marked as private so it's not included in the final agent state.
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """State update for `MemoryMiddleware`."""

    memory_contents: dict[str, str]


MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    上方 <agent_memory> 从文件系统加载。通过与用户交互学到新知识时，用 `edit_file` 工具保存。

    **学习反馈：**
    - 核心优先级之一是从用户交互中学习（显式或隐式）
    - 需要记住某事时，必须第一时间立即更新记忆，先于其他一切操作
    - 用户说某种分析方法好/坏时，记录原因并提炼为规律
    - 每次纠正都是永久改进的机会，不只修当前问题，要更新指令
    - 用户中断工具调用并给出反馈时，立即更新记忆再修改调用
    - 用户未明确要求记忆，但提供了对未来有用的信息时，也要主动更新

    **何时更新记忆：**
    - 用户明确要求记住（"记住这个偏好"、"以后都这样做"）
    - 用户描述你的角色或行为方式（"你是数据分析师"、"总是先做X"）
    - 用户对分析报告给出反馈——记录哪里错了、如何改进
    - 用户提供数据源配置、常用查询条件等未来任务有用的信息
    - 发现新的分析规律或用户偏好（报告格式、图表类型、关注重点）

    **何时不更新记忆：**
    - 信息是临时的（"今天加班"、"现在在手机上"）
    - 一次性任务请求（"查一下昨天的数据"、"这个数字是多少"）
    - 不揭示持久偏好的简单问题（"今天几号"、"解释一下这个字段"）
    - 认可或闲聊（"好的"、"谢谢"、"辛苦了"）
    - 信息在未来对话中已过时或不相关
    - 绝不存储 API 密钥、访问令牌、密码等凭证

    **示例：**
    示例 1（记住用户偏好）：
    用户：以后周报都按部门分组，不要按分类
    智能体：让我保存到记忆中。
    工具调用：edit_file(...) -> 记住用户偏好周报按部门分组

    示例 2（记住分析反馈）：
    用户：这个图表用柱状图更清楚
    智能体：让我保存到记忆中。
    工具调用：edit_file(...) -> 记住用户偏好用柱状图展示此类数据

    示例 3（不记住临时信息）：
    用户：今天下午开会，晚点再看报告
    智能体：好的，我先生成报告。
    （不调用 edit_file，因为是临时信息）
</memory_guidelines>
"""


class MemoryMiddleware(AgentMiddleware[MemoryState, ContextT, ResponseT]):
    """Middleware for loading agent memory from `AGENTS.md` files.

    Loads memory content from configured sources and injects into the system prompt.

    Supports multiple sources that are combined together.

    Args:
        backend: Backend instance or factory function for file operations.
        sources: List of `MemorySource` configurations specifying paths and names.
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        """Initialize the memory middleware.

        Args:
            backend: Backend instance or factory function that takes runtime
                     and returns a backend. Use a factory for StateBackend.
            sources: List of memory file paths to load (e.g., `["~/.deepagents/AGENTS.md",
                     "./.deepagents/AGENTS.md"]`).

                     Display names are automatically derived from the paths.

                     Sources are loaded in order.
        """
        self._backend = backend
        self.sources = sources

    def _get_backend(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            # Construct an artificial tool runtime to resolve backend factory
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
        return self._backend

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """Format memory with locations and contents paired together.

        Args:
            contents: Dict mapping source paths to content.

        Returns:
            Formatted string with location+content pairs wrapped in <agent_memory> tags.
        """
        if not contents:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        sections = [f"{path}\n{contents[path]}" for path in self.sources if contents.get(path)]

        if not sections:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        memory_body = "\n\n".join(sections)
        return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load memory content before agent execution (synchronous).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # Skip if already loaded
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        results = backend.download_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load memory content before agent execution.

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # Skip if already loaded
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        results = await backend.adownload_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject memory content into the system message.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with memory injected into system message.
        """
        contents = request.state.get("memory_contents", {})
        agent_memory = self._format_agent_memory(contents)

        new_system_message = append_to_system_message(request.system_message, agent_memory)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
