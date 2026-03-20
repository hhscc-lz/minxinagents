"""Unit tests for the welcome banner widget."""

from unittest.mock import MagicMock, patch

from rich.style import Style
from rich.text import Text

from deepagents_cli.widgets.welcome import (
    WelcomeBanner,
    build_connecting_footer,
    build_failure_footer,
    build_welcome_footer,
)

_WEBBROWSER_OPEN = "deepagents_cli.widgets._links.webbrowser.open"


def _make_banner(
    thread_id: str | None = None,
    *,
    connecting: bool = False,
) -> WelcomeBanner:
    """Create a `WelcomeBanner` instance for testing."""
    with patch.dict("os.environ", {}, clear=True):
        return WelcomeBanner(thread_id=thread_id, connecting=connecting)


class TestWelcomeBanner:
    """Tests for the simplified welcome banner."""

    def test_build_banner_shows_only_brand_and_welcome_text(self) -> None:
        """Banner should not include thread, LangSmith, or MCP details."""
        widget = _make_banner(thread_id="thread-123")
        banner = widget._build_banner()

        assert isinstance(banner, Text)
        assert "民心智能体已就绪，请问有什么可以帮您？" in banner.plain
        assert "参考示例：" in banner.plain
        assert "生成今天的全省热线日报" in banner.plain
        assert "LangSmith" not in banner.plain
        assert "会话：" not in banner.plain
        assert "MCP" not in banner.plain

    def test_update_thread_id_does_not_expose_thread_text(self) -> None:
        """Updating the thread should re-render without exposing thread info."""
        widget = _make_banner(thread_id="old-id")

        with patch.object(widget, "update") as mock_update:
            widget.update_thread_id("new-id")

        mock_update.assert_called_once()
        banner = mock_update.call_args[0][0]
        assert "会话：" not in banner.plain
        assert "old-id" not in banner.plain
        assert "new-id" not in banner.plain

    def test_connecting_state_uses_chinese_loading_text(self) -> None:
        """Connecting state should display the localized loading message."""
        widget = _make_banner(connecting=True)
        banner = widget._build_banner()
        assert "加载中..." in banner.plain

    def test_failed_state_uses_chinese_loading_failure_text(self) -> None:
        """Failed state should display the localized failure message."""
        widget = _make_banner()
        with patch.object(widget, "update"):
            widget.set_failed("网络异常")
        banner = widget._build_banner()
        assert "加载失败：" in banner.plain
        assert "网络异常" in banner.plain


class TestWelcomeFooterHelpers:
    """Tests for footer helper functions."""

    def test_build_welcome_footer_contains_ready_text(self) -> None:
        """Welcome footer should contain the ready prompt."""
        footer = build_welcome_footer()
        assert isinstance(footer, Text)
        assert "民心智能体已就绪，请问有什么可以帮您？" in footer.plain
        assert "参考示例：" in footer.plain
        assert "分析本周沈阳市物业投诉热点" in footer.plain

    def test_build_connecting_footer_contains_loading_text(self) -> None:
        """Connecting footer should use the Chinese loading text."""
        footer = build_connecting_footer()
        assert isinstance(footer, Text)
        assert "加载中..." in footer.plain

    def test_build_failure_footer_contains_loading_failed_text(self) -> None:
        """Failure footer should use the Chinese loading failure text."""
        footer = build_failure_footer("权限不足")
        assert isinstance(footer, Text)
        assert "加载失败：" in footer.plain
        assert "权限不足" in footer.plain


class TestAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`WelcomeBanner` should disable Textual's `auto_links`."""
        assert WelcomeBanner.auto_links is False


class TestOnClickOpensLink:
    """Tests for `WelcomeBanner.on_click` opening Rich-style hyperlinks."""

    def test_click_on_link_opens_browser(self) -> None:
        """Clicking a Rich link should call `webbrowser.open`."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_called_once_with("https://example.com")
        event.stop.assert_called_once()

    def test_click_without_link_is_noop(self) -> None:
        """Clicking on non-link text should not open the browser."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style()

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()

    def test_click_with_browser_error_is_graceful(self) -> None:
        """Browser failure should not crash the widget."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN, side_effect=OSError("no display")):
            widget.on_click(event)  # should not raise

        event.stop.assert_not_called()
