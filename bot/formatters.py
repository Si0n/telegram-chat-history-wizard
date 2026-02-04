"""
Format search results for Telegram display.
"""
from typing import Optional
from db.models import Message


class MessageFormatter:
    """Format messages and search results for Telegram."""

    MAX_MESSAGE_LENGTH = 4000  # Telegram limit is 4096

    @staticmethod
    def format_quote(
        text: str,
        username: str,
        date: str,
        similarity: float = None
    ) -> str:
        """Format a single message as a quote."""
        # Truncate long messages
        if len(text) > 500:
            text = text[:500] + "..."

        lines = [
            f"📅 {date} | {username}"
        ]

        if similarity is not None:
            pct = int(similarity * 100)
            lines[0] += f" ({pct}% match)"

        lines.append(f'"{text}"')

        return "\n".join(lines)

    @staticmethod
    def format_search_results(
        query: str,
        results: list[dict],
        page: int = 1,
        total_pages: int = 1
    ) -> str:
        """Format search results for display."""
        if not results:
            return f"🔍 Пошук: \"{query}\"\n\nНічого не знайдено."

        lines = [
            f"🔍 Пошук: \"{query}\"",
            f"Знайдено {len(results)} повідомлень",
            "━" * 30
        ]

        for result in results:
            meta = result.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            date = meta.get("formatted_date", "Unknown date")
            text = result.get("text", "")
            similarity = result.get("similarity")

            lines.append("")
            lines.append(MessageFormatter.format_quote(
                text=text,
                username=username,
                date=date,
                similarity=similarity
            ))

        if total_pages > 1:
            lines.append("")
            lines.append(f"[{page}/{total_pages}] Сторінка")

        return "\n".join(lines)

    @staticmethod
    def format_flip_result(result) -> str:
        """Format flip detection result."""
        if not result.messages:
            return f"❌ Не знайдено повідомлень від {result.user} про \"{result.topic}\""

        lines = [
            f"🔄 Аналіз позиції: {result.user}",
            f"📋 Тема: \"{result.topic}\"",
            "━" * 30
        ]

        if result.has_flip:
            lines.append("")
            lines.append(f"⚠️ ВИЯВЛЕНО ЗМІНУ ПОЗИЦІЇ!")
            lines.append(f"Впевненість: {result.confidence.upper()}")
        else:
            lines.append("")
            lines.append("✅ Значних змін позиції не виявлено")

        lines.append("")
        lines.append(f"📝 {result.summary}")

        # Show relevant messages
        lines.append("")
        lines.append("📜 Релевантні повідомлення:")
        lines.append("━" * 30)

        for msg in result.messages[:5]:  # Limit to 5 messages
            meta = msg.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            date = meta.get("formatted_date", "Unknown date")
            text = msg.get("text", "")[:300]

            lines.append("")
            lines.append(f"📅 {date} | {username}")
            lines.append(f'"{text}"')

        return "\n".join(lines)

    @staticmethod
    def format_stats(stats: dict) -> str:
        """Format database statistics."""
        lines = [
            "📊 Статистика бази даних",
            "━" * 30,
            "",
            f"📨 Всього повідомлень: {stats.get('total_messages', 0):,}",
            f"🔍 Проіндексовано: {stats.get('embedded_messages', 0):,}",
            f"👥 Унікальних користувачів: {stats.get('unique_users', 0):,}",
            f"📦 Експортів оброблено: {stats.get('exports_count', 0)}",
        ]

        if stats.get("date_start") and stats.get("date_end"):
            start = stats["date_start"].strftime("%d.%m.%Y")
            end = stats["date_end"].strftime("%d.%m.%Y")
            lines.append(f"📅 Період: {start} — {end}")

        return "\n".join(lines)

    @staticmethod
    def format_context(
        target_msg: Message,
        context_messages: list[Message],
        search_query: str = None
    ) -> str:
        """Format conversation context around a message with optional highlighting."""
        lines = [
            f"💬 Контекст повідомлення #{target_msg.message_id}",
        ]

        if search_query:
            lines.append(f"🔍 Запит: {search_query}")

        lines.append("━" * 30)

        # Extract keywords for highlighting (words > 3 chars)
        highlight_words = []
        if search_query:
            highlight_words = [
                w.lower() for w in search_query.split()
                if len(w) > 3 and w.isalpha()
            ]

        for msg in context_messages:
            is_target = msg.message_id == target_msg.message_id
            marker = "▶️" if is_target else "  "
            lines.append("")
            lines.append(f"{marker} 📅 {msg.formatted_date} | {msg.display_name}")

            text = msg.text[:600] if msg.text else "[пусто]"

            # Highlight relevant words in target message
            if is_target and highlight_words:
                text = MessageFormatter._highlight_text(text, highlight_words)

            lines.append(f'   "{text}"')

        return "\n".join(lines)

    @staticmethod
    def _highlight_text(text: str, keywords: list[str]) -> str:
        """Highlight keywords in text using «» markers."""
        import re
        result = text
        for keyword in keywords:
            # Case-insensitive replacement with markers
            pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
            result = pattern.sub(r'«\1»', result)
        return result

    @staticmethod
    def format_help() -> str:
        """Format help message."""
        return """🤖 Бот пошуку по історії чату

📋 Команди:

/search <запит> — Семантичний пошук по всій історії
  Приклад: /search криптовалюти інвестиції

/quote @username <тема> — Що казав користувач про тему
  Приклад: /quote @ivan_petrov біткоін

/flip @username <тема> — Перевірити чи змінював позицію
  Приклад: /flip @ivan_petrov крипта

/context <message_id> — Показати контекст повідомлення

/stats — Статистика бази даних

/upload — Відповісти на .zip файл щоб додати новий експорт

📝 Прізвиська:

/aliases — Список всіх прізвиськ
/alias <username> <прізвисько> — Додати прізвисько
/alias_remove <прізвисько> — Видалити прізвисько
/seed_aliases — Налаштувати стандартні прізвиська

🤖 AI запитання:

Просто тегни бота з питанням:
@dobby_the_free_trader_bot чи гусь казав що біткоін буде рости?

Бот розуміє прізвиська та шукає в історії чату.
Відповідай на результати для уточнень.

💡 Поради:
• Пошук працює семантично — знайде схожі за змістом
• Можна використовувати прізвиська замість імен
• Відповідай на результати бота для уточнення пошуку"""

    @staticmethod
    def format_upload_success(stats: dict) -> str:
        """Format successful upload message."""
        return f"""✅ Експорт успішно завантажено!

📁 Папка: {stats.get('export_path', 'N/A')}
💬 Чат: {stats.get('chat_name', 'N/A')}
📨 Повідомлень: ~{stats.get('estimated_messages', 0):,}

⏳ Індексація почнеться автоматично..."""

    @staticmethod
    def format_upload_error(error: str) -> str:
        """Format upload error message."""
        return f"❌ Помилка завантаження:\n{error}"

    @staticmethod
    def format_ai_response(
        question: str,
        results: list[dict],
        mentioned_users: list[tuple[int, str]] = None
    ) -> str:
        """Format AI question response with quotes."""
        if not results:
            user_str = ""
            if mentioned_users:
                names = [u[1] for u in mentioned_users]
                user_str = f" від {', '.join(names)}"
            return f"🔍 Питання: \"{question}\"\n\n❌ Не знайдено повідомлень{user_str}."

        lines = [
            f"🔍 \"{question}\"",
            f"📝 Знайдено {len(results)} повідомлень:",
            "━" * 30
        ]

        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            date = meta.get("formatted_date", "Unknown date")
            text = result.get("text", "")
            msg_id = meta.get("message_id", "")

            # Truncate long messages
            if len(text) > 400:
                text = text[:400] + "..."

            lines.append("")
            lines.append(f"👤 {username}")
            lines.append(f"📅 {date}")
            lines.append(f"> {text}")
            if msg_id:
                lines.append(f"🔗 /context {msg_id}")

        lines.append("")
        lines.append("━" * 30)
        lines.append("💡 Відповідай на це повідомлення для уточнення")

        return "\n".join(lines)

    @staticmethod
    def format_aliases(aliases: list, users: list[tuple[int, str]]) -> str:
        """Format aliases list."""
        if not aliases:
            return "📋 Прізвиська ще не налаштовані.\n\nВикористовуйте /alias @username прізвисько"

        # Group by user
        user_aliases = {}
        for alias in aliases:
            uid = alias.user_id
            if uid not in user_aliases:
                user_aliases[uid] = {"username": alias.username, "aliases": []}
            user_aliases[uid]["aliases"].append(alias.alias)

        lines = ["📋 Прізвиська користувачів:", "━" * 30]

        for uid, info in user_aliases.items():
            aliases_str = ", ".join(info["aliases"])
            lines.append(f"\n👤 {info['username']}")
            lines.append(f"   └ {aliases_str}")

        return "\n".join(lines)

    @staticmethod
    def format_synthesized_answer(
        question: str,
        synthesized,
        mentioned_users: list[tuple[int, str]] = None,
        date_from: str = None,
        date_to: str = None
    ) -> str:
        """Format AI-synthesized answer with supporting quotes."""
        lines = [
            f"❓ {question}",
        ]

        # Show date filter if applied
        if date_from or date_to:
            date_str = ""
            if date_from and date_to:
                date_str = f"📅 Період: {date_from} — {date_to}"
            elif date_from:
                date_str = f"📅 Після: {date_from}"
            elif date_to:
                date_str = f"📅 До: {date_to}"
            lines.append(date_str)

        lines.extend([
            "━" * 30,
            "",
            f"🤖 {synthesized.answer}",
        ])

        # Add supporting quotes if available
        if synthesized.supporting_quotes:
            lines.append("")
            lines.append("━" * 30)
            lines.append("📜 Підтверджуючі цитати:")

            for i, result in enumerate(synthesized.supporting_quotes[:3], 1):
                meta = result.get("metadata", {})
                username = meta.get("display_name", "Unknown")
                date = meta.get("formatted_date", "Unknown")
                text = result.get("text", "")

                # Show more text (up to 800 chars)
                if len(text) > 800:
                    text = text[:800] + "..."

                lines.append("")
                lines.append(f"[{i}] 👤 {username} | 📅 {date}")
                lines.append(f"> {text}")

        lines.append("")
        lines.append("━" * 30)
        lines.append("💡 Відповідай на це повідомлення для уточнення")

        return "\n".join(lines)
