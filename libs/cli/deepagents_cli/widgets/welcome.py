"""Welcome banner widget for deepagents-cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click

from deepagents_cli.config import (
    COLORS,
    _is_editable_install,
    get_banner,
)
from deepagents_cli.widgets._links import open_style_link


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        background: #0b151b;
        border: solid #1d3b37;
        color: #d7efe9;
    }
    """

    def __init__(
        self,
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        connecting: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Optional thread ID to display in the banner.
            mcp_tool_count: Number of MCP tools loaded at startup.
            connecting: When `True`, show a "Connecting..." footer instead of
                the normal ready prompt. Call `set_connected` to transition.
            **kwargs: Additional arguments passed to parent.
        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._connecting = connecting
        self._failed = False
        self._failure_error: str = ""

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """No-op mount hook for symmetry with other widgets."""

    def update_thread_id(self, thread_id: str) -> None:
        """Update the displayed thread ID and re-render the banner.

        Args:
            thread_id: The new thread ID to display.
        """
        self._cli_thread_id = thread_id
        self.update(self._build_banner())

    def set_connected(self, mcp_tool_count: int = 0) -> None:
        """Transition from "connecting" to "ready" state.

        Args:
            mcp_tool_count: Number of MCP tools loaded during connection.
        """
        self._connecting = False
        self._failed = False
        self._mcp_tool_count = mcp_tool_count
        self.update(self._build_banner())

    def set_failed(self, error: str) -> None:
        """Transition from "connecting" to a persistent failure state.

        Args:
            error: Error message describing the server startup failure.
        """
        self._connecting = False
        self._failed = True
        self._failure_error = error
        self.update(self._build_banner())

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open Rich-style hyperlinks on single click."""
        open_style_link(event)

    def _build_banner(self, project_url: str | None = None) -> Text:
        """Build the banner rich text.

        Args:
            project_url: Unused legacy parameter kept for compatibility.

        Returns:
            Rich Text object containing the formatted banner.
        """
        del project_url
        banner = Text()
        # Use orange for local, green for production
        banner_color = (
            COLORS["primary_dev"] if _is_editable_install() else COLORS["primary"]
        )
        banner.append(get_banner() + "\n", style=Style(bold=True, color=banner_color))

        if self._failed:
            banner.append_text(build_failure_footer(self._failure_error))
        elif self._connecting:
            banner.append_text(build_connecting_footer())
        else:
            banner.append_text(build_welcome_footer())
        return banner


def build_failure_footer(error: str) -> Text:
    """Build a footer shown when the server failed to start.

    Args:
        error: Error message describing the failure.

    Returns:
        Rich Text with a persistent failure message.
    """
    footer = Text()
    footer.append("\n加载失败：", style="bold red")
    footer.append(error, style="red")
    footer.append("\n", style="red")
    return footer


def build_connecting_footer() -> Text:
    """Build a footer shown while waiting for the server to connect.

    Returns:
        Rich Text with a connecting status message.
    """
    footer = Text()
    footer.append("\n加载中...\n", style="dim")
    return footer


def build_welcome_footer() -> Text:
    """Build the footer shown at the bottom of the welcome banner.

    Returns:
        Rich Text with the ready prompt.
    """
    footer = Text()
    footer.append(
        "\n民心智能体已就绪，请问有什么可以帮您？\n", style=COLORS["primary"]
    )
    footer.append("参考示例：\n", style="bold #d7efe9")
    footer.append("• 生成今天的全省热线日报\n", style="dim")
    footer.append("• 分析本周沈阳市物业投诉热点\n", style="dim")
    footer.append("• 排查最近三天铁西区是否有欠薪风险\n", style="dim")
    footer.append("• 按区县比较大连市供暖诉求变化\n", style="dim")
    footer.append("• 帮我整理一个领导汇报专题\n", style="dim")
    return footer
