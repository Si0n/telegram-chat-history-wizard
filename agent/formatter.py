import re
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import config


class Formatter:
    """Format search results and dialogue windows for Telegram."""

    NUMBER_EMOJIS = ["1\u20e3", "2\u20e3", "3\u20e3"]

    def escape_html(self, text: str) -> str:
        """Escape HTML special characters for Telegram HTML mode."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def highlight(self, text: str, terms: list[str]) -> str:
        """Bold highlight terms in already-escaped HTML text (case-insensitive)."""
        for term in sorted(terms, key=len, reverse=True):
            escaped_term = self.escape_html(term)
            pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<b>{m.group()}</b>", text)
        return text

    def truncate_html(self, text: str, max_chars: int) -> str:
        """Truncate HTML-escaped text at max_chars of visible content, preserving tags."""
        visible = 0
        result = []
        open_tags = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                # Track HTML tags (don't count towards visible chars)
                end = text.find(">", i)
                if end == -1:
                    break
                tag = text[i : end + 1]
                result.append(tag)
                if tag.startswith("</"):
                    if open_tags:
                        open_tags.pop()
                elif not tag.endswith("/>"):
                    open_tags.append(tag)
                i = end + 1
            elif text[i] == "&":
                # HTML entity counts as 1 visible char
                end = text.find(";", i)
                if end == -1:
                    break
                result.append(text[i : end + 1])
                visible += 1
                i = end + 1
            else:
                result.append(text[i])
                visible += 1
                i += 1
            if visible >= max_chars and i < len(text):
                result.append("...")
                break
        # Close any open tags
        for tag in reversed(open_tags):
            tag_name = tag[1:].split()[0].rstrip(">")
            result.append(f"</{tag_name}>")
        return "".join(result)

    def _format_timestamp(self, ts_str: str) -> str:
        """Format ISO timestamp to DD.MM.YYYY HH:MM."""
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.strftime("%d.%m.%Y %H:%M")
        except (ValueError, TypeError):
            return "Unknown date"

    def _char_budget_per_message(self, msg_count: int) -> int:
        """Calculate per-message character budget to stay within Telegram limit."""
        overhead = 80  # header line
        per_msg_overhead = 50  # name + date line per message
        available = config.MESSAGE_CHAR_LIMIT - overhead - (per_msg_overhead * msg_count)
        return max(available // max(msg_count, 1), 100)

    def format_search_results(
        self,
        page_results: list[dict],
        total: int,
        page: int,
        highlight_terms: list[str],
        sort_order: str,
        explanation: str = "",
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        """Format a page of search results with inline keyboard."""
        if not page_results:
            return "Nothing found. Try rephrasing your question.", None

        per_page = config.RESULTS_PER_PAGE
        start = page * per_page + 1
        end = start + len(page_results) - 1
        sort_label = "oldest first" if sort_order == "asc" else "newest first"

        lines = []
        if explanation:
            lines.append(f"{self.escape_html(explanation)}\n")
        lines.append(f"Found {total} messages. Showing {start}-{end} ({sort_label}):\n")
        budget = self._char_budget_per_message(len(page_results))

        for i, msg in enumerate(page_results):
            name = self.escape_html(msg.get("first_name") or msg.get("username") or "Unknown")
            date = self._format_timestamp(msg.get("timestamp", ""))
            text = msg.get("text") or ""
            text = self.escape_html(text)
            text = self.highlight(text, highlight_terms)
            text = self.truncate_html(text, budget)

            emoji = self.NUMBER_EMOJIS[i] if i < len(self.NUMBER_EMOJIS) else f"{i+1}."
            lines.append(f"{emoji} <b>{name}</b> \u00b7 {date}")
            lines.append(text)
            lines.append("")

        # Build keyboard
        buttons_nav = []
        total_pages = (total + per_page - 1) // per_page
        if page > 0:
            buttons_nav.append(InlineKeyboardButton("\u2b05\ufe0f Prev", callback_data=f"p:{page - 1}"))
        if page < total_pages - 1:
            buttons_nav.append(InlineKeyboardButton("Next \u27a1\ufe0f", callback_data=f"p:{page + 1}"))

        buttons_dialogue = []
        for i, msg in enumerate(page_results):
            buttons_dialogue.append(
                InlineKeyboardButton(f"\U0001f4ac {i+1}", callback_data=f"d:{msg['id']}")
            )

        keyboard_rows = []
        if buttons_nav:
            keyboard_rows.append(buttons_nav)
        if buttons_dialogue:
            keyboard_rows.append(buttons_dialogue)

        keyboard = InlineKeyboardMarkup(keyboard_rows) if keyboard_rows else None
        return "\n".join(lines).strip(), keyboard

    def format_dialogue_window(
        self,
        messages: list[dict],
        anchor_id: int,
        highlight_terms: list[str],
        has_earlier: bool,
        has_later: bool,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Format a dialogue window around an anchor message."""
        budget = self._char_budget_per_message(len(messages))

        lines = []
        for msg in messages:
            name = self.escape_html(msg.get("first_name") or msg.get("username") or "Unknown")
            date = self._format_timestamp(msg.get("timestamp", ""))
            text = msg.get("text") or ""

            is_anchor = msg.get("id") == anchor_id
            if is_anchor:
                text = self.escape_html(text)
                text = self.highlight(text, highlight_terms)
                text = self.truncate_html(text, budget)
                lines.append(f"\U0001f449 <b>{name}</b> \u00b7 {date}")
            else:
                text = self.escape_html(text)
                text = self.truncate_html(text, budget)
                lines.append(f"{name} \u00b7 {date}")
            lines.append(text)
            lines.append("")

        # Navigation buttons
        buttons_nav = []
        if has_earlier:
            first_ts = messages[0].get("timestamp_unix") or 0
            buttons_nav.append(InlineKeyboardButton("\u2b05\ufe0f Back", callback_data=f"db:{first_ts}"))
        if has_later:
            last_ts = messages[-1].get("timestamp_unix") or 0
            buttons_nav.append(InlineKeyboardButton("Forward \u27a1\ufe0f", callback_data=f"df:{last_ts}"))

        buttons_back = [InlineKeyboardButton("\U0001f519 Back to results", callback_data="br")]

        keyboard_rows = []
        if buttons_nav:
            keyboard_rows.append(buttons_nav)
        keyboard_rows.append(buttons_back)

        return "\n".join(lines).strip(), InlineKeyboardMarkup(keyboard_rows)
