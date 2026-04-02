"""Deep Agents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents._models import resolve_model
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import create_summarization_middleware

# BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.
#
# ## Core Behavior
#
# - Be concise and direct. Don't over-explain unless asked.
# - NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
# - Don't say \"I'll now do X\" — just do it.
# - If the request is ambiguous, ask questions before acting.
# - If asked how to approach something, explain first, then act.
#
# ## Professional Objectivity
#
# - Prioritize accuracy over validating the user's beliefs
# - Disagree respectfully when the user is incorrect
# - Avoid unnecessary superlatives, praise, or emotional validation
#
# ## Doing Tasks
#
# When the user asks you to do something:
#
# 1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
# 2. **Act** — implement the solution. Work quickly but accurately.
# 3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.
#
# Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.
#
# **When things go wrong:**
# - If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
# - If you're blocked, tell the user what's wrong and ask for guidance.
#
# ## Progress Updates
#
# For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""  # noqa: E501

BASE_AGENT_PROMPT = ""

MINXIN_TODOS_SYSTEM_PROMPT = """## `write_todos`

使用 `write_todos` 工具管理和规划复杂目标。
对复杂目标使用此工具，确保跟踪每个必要步骤并让用户了解进度。
此工具对规划复杂目标、将大型复杂目标分解为小步骤非常有用。

完成步骤后立即标记为已完成至关重要。不要在标记完成前批量处理多个步骤。
对于只需几步的简单目标，最好直接完成目标，不使用此工具。
编写任务需要时间和 token，在管理复杂多步骤问题时使用！但不用于简单的几步请求。

## 重要的任务列表使用注意事项
- `write_todos` 工具不应并行调用多次
- 不要害怕随时修订任务列表。新信息可能揭示需要完成的新任务，或使旧任务无关"""

MINXIN_TODOS_TOOL_DESCRIPTION = """使用此工具创建和管理当前工作会话的结构化任务列表。帮助跟踪进度、组织复杂任务、向用户展示工作全面性。

仅在认为有助于保持组织性时使用。如果用户请求简单且少于3步，最好不使用此工具，直接完成任务。

## 何时使用此工具
以下场景使用：

1. 复杂多步骤任务 - 需要3个或更多不同步骤或操作
2. 非平凡的复杂任务 - 需要仔细规划或多个操作
3. 用户明确要求任务列表 - 用户直接要求使用任务列表
4. 用户提供多个任务 - 用户提供任务列表（编号或逗号分隔）
5. 计划可能需要根据前几步结果进行修订或更新

## 如何使用此工具
1. 开始任务时 - 开始工作前标记为 in_progress
2. 完成任务后 - 标记为 completed，添加实施过程中发现的新后续任务
3. 可更新未来任务，如删除不再必要的任务，或添加必要的新任务。不要更改已完成的任务
4. 可一次对任务列表进行多次更新。例如完成任务时，可将下一个需要开始的任务标记为 in_progress

## 何时不使用此工具
以下情况跳过：
1. 只有单个直接任务
2. 任务简单，跟踪无益处
3. 任务可在少于3个简单步骤内完成
4. 任务纯粹是对话或信息性的

## 任务状态和管理

1. **任务状态**：
   - pending：尚未开始
   - in_progress：正在进行（不相关且可并行的任务可同时有多个 in_progress）
   - completed：成功完成

2. **任务管理**：
   - 工作时实时更新任务状态
   - 完成后立即标记（不要批量标记）
   - 从列表中完全删除不再相关的任务
   - 重要：编写任务列表时，应立即将第一个任务标记为 in_progress
   - 重要：除非所有任务完成，否则应始终至少有一个 in_progress 任务

3. **任务完成要求**：
   - 仅在完全完成时才标记为 completed
   - 遇到错误、阻碍或无法完成时，保持 in_progress
   - 被阻塞时，创建描述需要解决问题的新任务
   - 以下情况绝不标记为 completed：有未解决问题、工作部分完成、遇到阻碍、找不到必要资源

4. **任务分解**：
   - 创建具体、可操作的项目
   - 将复杂任务分解为更小、可管理的步骤
   - 使用清晰、描述性的任务名称

记住：如果只需几次工具调用即可完成任务且清楚要做什么，最好直接完成任务，完全不调用此工具。"""

MINXIN_FILESYSTEM_PROMPT = """## 文件工具

- `read_file`：读取文件内容（使用绝对路径）
- `write_file`：创建或覆盖文件（使用绝对路径）
- `edit_file`：精确替换文件中的字符串（必须先读取）

## Execute 工具

使用 `execute` 工具执行 shell 命令，用于运行 Python 脚本、数据处理、生成报告文件等操作。

- execute：执行 shell 命令（返回输出和退出码）"""


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.6.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-6",
    )


def create_deep_agent(  # noqa: C901, PLR0912  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    !!! warning "Deep agents require a LLM that supports tool calling!"

    By default, this agent has access to the following tools:

    - `write_todos`: manage a todo list
    - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: file operations
    - `execute`: run shell commands
    - `task`: call subagents

    The `execute` tool allows running shell commands if the backend implements `SandboxBackendProtocol`.
    For non-sandbox backends, the `execute` tool will return an error message.

    Args:
        model: The model to use.

            Defaults to `claude-sonnet-4-6`.

            Use the `provider:model` format (e.g., `openai:gpt-5`) to quickly switch between models.

            If an `openai:` model is used, the agent will use the OpenAI
            Responses API by default. To use OpenAI chat completions instead,
            initialize the model with
            `init_chat_model("openai:...", use_responses_api=False)` and pass
            the initialized model instance here. To disable data retention with
            the Responses API, use
            `init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
            and pass the initialized model instance here.
        tools: The tools the agent should have access to.

            In addition to custom tools you provide, deep agents include built-in tools for planning,
            file management, and subagent spawning.
        system_prompt: Custom system instructions to prepend before the base deep agent
            prompt.

            If a string, it's concatenated with the base prompt.
        middleware: Additional middleware to apply after the standard middleware stack
            (`TodoListMiddleware`, `FilesystemMiddleware`, `SubAgentMiddleware`,
            `SummarizationMiddleware`, `AnthropicPromptCachingMiddleware`,
            `PatchToolCallsMiddleware`).
        subagents: The subagents to use.

            Each subagent should be a `dict` with the following keys:

            - `name`
            - `description` (used by the main agent to decide whether to call the sub agent)
            - `system_prompt` (used as the system prompt in the subagent)
            - (optional) `tools`
            - (optional) `model` (either a `LanguageModelLike` instance or `dict` settings)
            - (optional) `middleware` (list of `AgentMiddleware`)
        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes) and are relative
            to the backend's root. When using `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded from disk relative
            to the backend's `root_dir`. Later sources override earlier ones for skills with the
            same name (last one wins).
        memory: Optional list of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.

            Memory is loaded at agent startup and added into the system prompt.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional `Checkpointer` for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses `StoreBackend`).
        backend: Optional backend for file storage and execution.

            Pass either a `Backend` instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.

            Pass to pause agent execution at specified tool calls for human approval or modification.

            Example: `interrupt_on={"edit_file": True}` pauses before every edit.
        debug: Whether to enable debug mode. Passed through to `create_agent`.
        name: The name of the agent. Passed through to `create_agent`.
        cache: The cache to use for the agent. Passed through to `create_agent`.

    Returns:
        A configured deep agent.
    """
    model = get_default_model() if model is None else resolve_model(model)

    backend = backend if backend is not None else (StateBackend)

    # Build general-purpose subagent with default middleware stack
    gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        create_summarization_middleware(model, backend),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    if interrupt_on is not None:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    }

    # Process user-provided subagents to fill in defaults for model, tools, and middleware
    processed_subagents: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            # CompiledSubAgent - use as-is
            processed_subagents.append(spec)
        else:
            # SubAgent - fill in defaults and prepend base middleware
            subagent_model = spec.get("model", model)
            subagent_model = resolve_model(subagent_model)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                create_summarization_middleware(subagent_model, backend),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
                "middleware": subagent_middleware,
            }
            processed_subagents.append(processed_spec)

    if any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in processed_subagents):
        # If an agent with general purpose name already exists in subagents, then don't add it
        # This is how you overwrite/configure general purpose subagent
        all_subagents: list[SubAgent | CompiledSubAgent] = processed_subagents
    else:
        # Otherwise - add it!
        all_subagents = [general_purpose_spec, *processed_subagents]

    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(
            system_prompt=MINXIN_TODOS_SYSTEM_PROMPT,
            tool_description=MINXIN_TODOS_TOOL_DESCRIPTION,
        ),
    ]
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend, system_prompt=MINXIN_FILESYSTEM_PROMPT),
            # SubAgentMiddleware(
            #     backend=backend,
            #     subagents=all_subagents,
            # ),
            # create_summarization_middleware(model, backend),
            # AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore", ttl="1h"),
            PatchToolCallsMiddleware(),
        ]
    )

    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"}])
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config(
        {
            "recursion_limit": 1000,
            "metadata": {
                "ls_integration": "deepagents",
            },
        }
    )
