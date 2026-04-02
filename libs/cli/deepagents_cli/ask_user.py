"""Ask user middleware for interactive question-answering during agent execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from typing import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import InjectedToolCallId
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from pydantic import Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class Choice(TypedDict):
    """A single choice option for a multiple choice question."""

    value: Annotated[str, Field(description="The display label for this choice.")]


class Question(TypedDict):
    """A question to ask the user."""

    question: Annotated[str, Field(description="The question text to display.")]

    type: Annotated[
        Literal["text", "multiple_choice"],
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' for "
                "predefined options."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Choice],
            Field(
                description=(
                    "Options for multiple_choice questions. An 'Other' free-form "
                    "option is always appended automatically."
                )
            ),
        ]
    ]

    required: NotRequired[
        Annotated[
            bool,
            Field(
                description="Whether the user must answer. Defaults to true if omitted."
            ),
        ]
    ]


class AskUserRequest(TypedDict):
    """Request payload sent via interrupt when asking the user questions."""

    type: Literal["ask_user"]

    questions: list[Question]

    tool_call_id: str


class AskUserAnswered(TypedDict):
    """Widget result when the user submits answers."""

    type: Literal["answered"]
    """Discriminator tag, always `'answered'`."""

    answers: list[str]
    """User-provided answers, one per question."""


class AskUserCancelled(TypedDict):
    """Widget result when the user cancels the prompt."""

    type: Literal["cancelled"]
    """Discriminator tag, always `'cancelled'`."""


# Discriminated union for the ask_user widget Future result.
AskUserWidgetResult = AskUserAnswered | AskUserCancelled


ASK_USER_TOOL_DESCRIPTION = """在需要用户提供信息或澄清时，向用户提出一个或多个问题。

每个问题可以是：
- "text"：用户自由输入文本
- "multiple_choice"：用户从预设选项中选择（始终自动追加"其他"选项）

多选题需提供选项列表，用户可选择一项或通过"其他"输入自定义答案。

默认所有问题为必答。将 "required" 设为 false 可允许用户跳过该问题。

适用场景：
- 需要澄清模糊的需求
- 需要用户在多个有效方案中做选择
- 需要只有用户才能提供的信息
- 执行前需要确认方案

不适用场景：
- 简单的是否确认（直接用最佳判断执行）
- 自己能从上下文回答的问题
- 不影响结果的无关紧要决定"""  # noqa: E501

ASK_USER_SYSTEM_PROMPT = """## `ask_user`

当你需要用户提供信息或澄清时，可使用 `ask_user` 工具。
仅在无法从上下文中自行判断时才使用，不要滥用。

使用时：
- 问题要简洁具体
- 有明确选项时用多选，需要自由回答时用文本
- 将相关问题合并为一次调用，不要多次调用
- 不要问自己能从上下文回答的问题"""  # noqa: E501


def _validate_questions(questions: list[Question]) -> None:
    """Validate ask_user question structure before interrupting.

    Args:
        questions: Question definitions provided to the `ask_user` tool.

    Raises:
        ValueError: If the questions list or an individual question is invalid.
    """
    if not questions:
        msg = "ask_user requires at least one question"
        raise ValueError(msg)

    for q in questions:
        question_text = q.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            msg = "ask_user questions must have non-empty 'question' text"
            raise ValueError(msg)

        question_type = q.get("type")
        if question_type not in {"text", "multiple_choice"}:
            msg = f"unsupported ask_user question type: {question_type!r}"
            raise ValueError(msg)

        if question_type == "multiple_choice" and not q.get("choices"):
            msg = (
                f"multiple_choice question "
                f"{q.get('question')!r} requires a "
                f"non-empty 'choices' list"
            )
            raise ValueError(msg)

        if question_type == "text" and q.get("choices"):
            msg = f"text question {q.get('question')!r} must not define 'choices'"
            raise ValueError(msg)


def _parse_answers(
    response: object,
    questions: list[Question],
    tool_call_id: str,
) -> Command[Any]:
    """Parse an interrupt response into a `Command` with a `ToolMessage`.

    Supports explicit status signaling from the adapter:

    - `answered` (default): consume provided `answers`
    - `cancelled`: synthesize `(cancelled)` answers
    - `error`: synthesize `(error: ...)` answers

    Malformed payloads are converted into explicit error answers instead of
    silently defaulting to `(no answer)`.

    Args:
        response: Raw value returned by `interrupt()`.
        questions: The questions that were asked.
        tool_call_id: Originating tool call ID for the `ToolMessage`.

    Returns:
        `Command` containing a formatted `ToolMessage` with Q&A pairs.
    """
    status: str = "answered"
    error_text: str | None = None
    answers: list[str]
    if not isinstance(response, dict):
        logger.error(
            "ask_user received malformed resume payload "
            "(expected dict, got %s); returning explicit error answers",
            type(response).__name__,
        )
        answers = []
        status = "error"
        error_text = "invalid ask_user response payload"
    else:
        response_dict = cast("dict[str, Any]", response)
        response_status = response_dict.get("status")
        if isinstance(response_status, str):
            status = response_status

        if "answers" not in response_dict:
            if status == "answered":
                logger.error(
                    "ask_user received resume payload without 'answers'; "
                    "returning explicit error answers"
                )
                answers = []
                status = "error"
                error_text = "missing ask_user answers payload"
            else:
                answers = []
        else:
            raw_answers = response_dict["answers"]
            if isinstance(raw_answers, list):
                answers = [str(answer) for answer in raw_answers]
            else:
                logger.error(
                    "ask_user received non-list 'answers' payload (%s); "
                    "returning explicit error answers",
                    type(raw_answers).__name__,
                )
                answers = []
                status = "error"
                error_text = "invalid ask_user answers payload"

        if status == "error":
            response_error = response_dict.get("error")
            if isinstance(response_error, str) and response_error:
                error_text = response_error
        elif status == "cancelled":
            answers = ["(cancelled)" for _ in questions]
        elif status == "answered":
            if len(answers) != len(questions):
                logger.warning(
                    "ask_user answer count mismatch: expected %d, got %d",
                    len(questions),
                    len(answers),
                )
        else:
            logger.error(
                "ask_user received unknown status %r; returning explicit error answers",
                status,
            )
            answers = []
            status = "error"
            error_text = "invalid ask_user response status"

    if status == "error":
        detail = error_text or "ask_user interaction failed"
        answers = [f"(error: {detail})" for _ in questions]

    formatted_answers = []
    for i, q in enumerate(questions):
        answer = answers[i] if i < len(answers) else "(no answer)"
        formatted_answers.append(f"Q: {q['question']}\nA: {answer}")
    result_text = "\n\n".join(formatted_answers)
    return Command(
        update={
            "messages": [ToolMessage(result_text, tool_call_id=tool_call_id)],
        }
    )


class AskUserMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that provides an ask_user tool for interactive questioning.

    This middleware adds an `ask_user` tool that allows agents to ask the user
    questions during execution. Questions can be free-form text or multiple choice.
    The tool uses LangGraph interrupts to pause execution and wait for user input.
    """

    def __init__(
        self,
        *,
        system_prompt: str = ASK_USER_SYSTEM_PROMPT,
        tool_description: str = ASK_USER_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize AskUserMiddleware.

        Args:
            system_prompt: System-level instructions injected into every LLM
                request to guide `ask_user` usage.
            tool_description: Description string passed to the `ask_user` tool
                decorator, visible to the LLM in the tool schema.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        def _ask_user(
            questions: list[Question],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """Ask the user one or more questions.

            Args:
                questions: Questions to present to the user.
                tool_call_id: Tool call identifier injected by LangChain.

            Returns:
                `Command` containing the parsed user answers as a `ToolMessage`.
            """
            _validate_questions(questions)
            ask_request = AskUserRequest(
                type="ask_user",
                questions=questions,
                tool_call_id=tool_call_id,
            )
            response = interrupt(ask_request)
            return _parse_answers(response, questions, tool_call_id)

        _ask_user.name = "ask_user"
        self.tools = [_ask_user]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt.

        Returns:
            Model response from the wrapped handler.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt (async).

        Returns:
            Model response from the wrapped handler.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
