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
        similarity: float = None,
        is_forwarded: bool = False,
        forward_from: str = None
    ) -> str:
        """Format a single message as a quote."""
        # Truncate long messages
        if len(text) > 500:
            text = text[:500] + "..."

        # Build header based on whether it's a forward
        if is_forwarded and forward_from:
            lines = [
                f"↪️ 📅 {date} | {username} переслав від {forward_from}"
            ]
        elif is_forwarded:
            lines = [
                f"↪️ 📅 {date} | {username} (переслано)"
            ]
        else:
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
    def format_paginated_results(
        question: str,
        results: list[dict],
        current_page: int,
        total_pages: int,
        total_results: int
    ) -> str:
        """Format paginated search results."""
        if not results:
            return f"🔍 \"{question}\"\n\nНічого не знайдено."

        lines = [
            f"🔍 \"{question}\"",
            f"📊 Знайдено {total_results} повідомлень (стор. {current_page}/{total_pages})",
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

            # Calculate global index
            start_idx = (current_page - 1) * 5
            global_idx = start_idx + i

            lines.append("")
            lines.append(f"[{global_idx}] 👤 {username}")
            lines.append(f"📅 {date}")
            lines.append(f"> {text}")

        lines.append("")
        lines.append("━" * 30)
        lines.append("💡 Використовуй кнопки для навігації")

        return "\n".join(lines)

    @staticmethod
    def format_filtered_results(
        question: str,
        results: list[dict],
        filter_label: str = None,
        total_results: int = 0
    ) -> str:
        """Format search results with active time filter."""
        if not results:
            filter_str = f" ({filter_label})" if filter_label else ""
            return f"🔍 \"{question}\"{filter_str}\n\n❌ Немає повідомлень за цей період."

        lines = [
            f"🔍 \"{question}\"",
        ]

        filter_str = f"📅 {filter_label}" if filter_label else ""
        if filter_str:
            lines.append(filter_str)

        lines.append(f"📊 Знайдено {total_results} повідомлень")
        lines.append("━" * 30)

        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            date = meta.get("formatted_date", "Unknown date")
            text = result.get("text", "")

            if len(text) > 400:
                text = text[:400] + "..."

            lines.append("")
            lines.append(f"[{i}] 👤 {username}")
            lines.append(f"📅 {date}")
            lines.append(f"> {text}")

        lines.append("")
        lines.append("━" * 30)
        lines.append("💡 Використовуй кнопки для фільтрації")

        return "\n".join(lines)

    @staticmethod
    def format_user_stats(
        user_stats: list[dict],
        hourly_distribution: dict[int, int],
        total_messages: int = 0
    ) -> str:
        """Format user statistics dashboard."""
        lines = [
            "📊 Статистика користувачів",
            "━" * 30,
            ""
        ]

        if total_messages:
            lines.append(f"📨 Всього повідомлень: {total_messages:,}")
            lines.append("")

        # Top users by message count
        if user_stats:
            lines.append("🏆 Топ-10 за повідомленнями:")
            medals = ["🥇", "🥈", "🥉"]

            for i, user in enumerate(user_stats[:10], 1):
                medal = medals[i-1] if i <= 3 else f"{i}."
                username = user.get("display_name", "Unknown")
                count = user.get("message_count", 0)

                # Format date range
                first = user.get("first_message")
                last = user.get("last_message")
                date_range = ""
                if first and last:
                    try:
                        first_str = first.strftime("%d.%m.%y") if hasattr(first, 'strftime') else str(first)[:10]
                        last_str = last.strftime("%d.%m.%y") if hasattr(last, 'strftime') else str(last)[:10]
                        date_range = f" ({first_str} - {last_str})"
                    except Exception:
                        pass

                lines.append(f"{medal} @{username}: {count:,}{date_range}")

        # Hourly activity chart
        if hourly_distribution:
            lines.append("")
            lines.append("━" * 30)
            lines.append("⏰ Активність за годинами:")
            lines.append("")

            # Find max for scaling
            max_count = max(hourly_distribution.values()) if hourly_distribution.values() else 1

            # Group into 4-hour blocks for compact display
            blocks = [
                ("00-04", sum(hourly_distribution.get(h, 0) for h in range(0, 4))),
                ("04-08", sum(hourly_distribution.get(h, 0) for h in range(4, 8))),
                ("08-12", sum(hourly_distribution.get(h, 0) for h in range(8, 12))),
                ("12-16", sum(hourly_distribution.get(h, 0) for h in range(12, 16))),
                ("16-20", sum(hourly_distribution.get(h, 0) for h in range(16, 20))),
                ("20-24", sum(hourly_distribution.get(h, 0) for h in range(20, 24))),
            ]

            block_max = max(b[1] for b in blocks) if blocks else 1

            for label, count in blocks:
                bar_length = int((count / block_max) * 10) if block_max > 0 else 0
                bar = "█" * bar_length + "░" * (10 - bar_length)
                lines.append(f"{label}: {bar} {count:,}")

        return "\n".join(lines)

    @staticmethod
    def format_thread(
        thread_data: dict,
        summary: str = None,
        max_messages: int = 15
    ) -> str:
        """Format conversation thread display."""
        messages = thread_data.get("messages", [])
        participants = thread_data.get("participants", set())
        duration = thread_data.get("duration_minutes", 0)
        total_count = thread_data.get("message_count", len(messages))

        lines = [
            f"🧵 Розмова ({total_count} повідомлень)",
            f"👥 Учасники: {len(participants)}",
        ]

        if duration > 0:
            if duration < 60:
                lines.append(f"⏱️ Тривалість: {duration} хв")
            else:
                hours = duration // 60
                mins = duration % 60
                lines.append(f"⏱️ Тривалість: {hours} год {mins} хв")

        lines.append("━" * 30)

        # Add summary if provided
        if summary:
            lines.append("")
            lines.append("📝 Короткий зміст:")
            for point in summary.split("\n"):
                if point.strip():
                    lines.append(f"• {point.strip()}")
            lines.append("")
            lines.append("━" * 30)

        # Build reply tree structure
        reply_map = {}  # message_id -> reply_to_id
        for msg in messages:
            if msg.reply_to_message_id:
                reply_map[msg.message_id] = msg.reply_to_message_id

        # Display messages with indentation for replies
        displayed = 0
        for msg in messages:
            if displayed >= max_messages:
                remaining = total_count - displayed
                if remaining > 0:
                    lines.append(f"\n... ще {remaining} повідомлень")
                break

            # Determine indentation level based on reply chain
            indent = ""
            if msg.message_id in reply_map:
                indent = "└ "

            username = msg.username or f"User#{msg.user_id}" if msg.user_id else "Unknown"
            date = msg.formatted_date if hasattr(msg, 'formatted_date') else str(msg.timestamp)[:16]

            text = msg.text or "[медіа/стікер]"
            if len(text) > 300:
                text = text[:300] + "..."

            lines.append("")
            lines.append(f"{indent}👤 @{username} ({date})")
            lines.append(f"{indent}  {text}")

            displayed += 1

        return "\n".join(lines)

    @staticmethod
    def format_thread_summary_prompt(messages: list) -> str:
        """Format messages for thread summarization."""
        lines = []
        for msg in messages[:20]:  # Limit for API
            username = msg.username or f"User#{msg.user_id}" if msg.user_id else "Unknown"
            text = msg.text[:200] if msg.text else "[non-text]"
            lines.append(f"{username}: {text}")
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
            f"🔗 Пов'язані повідомлення ({len(context_messages)})",
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

💬 Як користуватись:

uТегни мене з питанням:
@dobby_the_free_trader_bot чи гусь казав що біткоін буде рости?
@dobby_the_free_trader_bot що думав буш про крипту до 2022?
@dobby_the_free_trader_bot покажи старі повідомлення спочатку

Відповідай на мої повідомлення для уточнень.

📅 Фільтри дат:
• "до 2022", "після березня 2023"
• "в період 2020-2021"

🔄 Сортування:
• "старі спочатку" / "нові спочатку"

📋 Команди:
/stats — Статистика бази
/mystats — Статистика користувачів
/aliases — Список прізвиськ користувачів
/alias @user прізвисько — Додати прізвисько
/alias_remove прізвисько — Видалити прізвисько

🔤 Аліаси сутностей (зе → Зеленський):
/entity_aliases — Список аліасів
/entity_alias <аліас> <канон> [категорія]
/entity_alias_remove <аліас>

💡 Поради:
• Використовуй прізвиська: гусь, буш, серж...
• Використовуй сленг: зе, порох, біток...
• Натисни кнопку "Контекст" для перегляду діалогу"""

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
    def format_synthesized_answer_header(
        question: str,
        date_from: str = None,
        date_to: str = None,
        sort_order: str = "relevance"
    ) -> str:
        """Format header for streamed answer (without the answer content)."""
        lines = [
            f"❓ {question}",
        ]

        # Show filters if applied
        filters = []
        if date_from and date_to:
            filters.append(f"📅 {date_from} — {date_to}")
        elif date_from:
            filters.append(f"📅 після {date_from}")
        elif date_to:
            filters.append(f"📅 до {date_to}")

        if sort_order == "oldest":
            filters.append("⬆️ старі спочатку")
        elif sort_order == "newest":
            filters.append("⬇️ нові спочатку")

        if filters:
            lines.append(" | ".join(filters))

        lines.extend([
            "━" * 30,
            "",
            "🤖 ",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_synthesized_answer_with_answer(
        header: str,
        answer: str,
        synthesized,
        quotes_with_context: list[dict] = None
    ) -> str:
        """Format complete answer combining header, streamed answer, and quotes with context."""
        lines = [header.rstrip() + answer]

        # Add supporting quotes if available
        if quotes_with_context:
            lines.append("")
            lines.append("━" * 30)
            lines.append("📜 Підтверджуючі цитати:")

            for item in quotes_with_context:
                quote = item["quote"]
                context_msgs = item.get("context", [])

                meta = quote.get("metadata", {})
                username = meta.get("display_name", "Unknown")
                date = meta.get("formatted_date", "Unknown")
                target_msg_id = meta.get("message_id")
                quote_text = quote.get("text", "")

                lines.append("")

                # If no context, show quote directly (cleaner for date-filtered results)
                if not context_msgs:
                    if len(quote_text) > 500:
                        quote_text = quote_text[:500] + "..."
                    lines.append(f"👤 {username} | 📅 {date}")
                    lines.append(f"> {quote_text}")
                else:
                    # Show context messages with the target highlighted
                    lines.append(f"💬 {username}:")
                    for msg in context_msgs:
                        msg_username = msg.display_name if hasattr(msg, 'display_name') else "Unknown"
                        msg_date = msg.formatted_date if hasattr(msg, 'formatted_date') else ""
                        msg_text = msg.text[:300] if hasattr(msg, 'text') and msg.text else ""
                        if len(msg_text) == 300:
                            msg_text += "..."

                        is_target = hasattr(msg, 'message_id') and msg.message_id == target_msg_id
                        marker = "▶️" if is_target else "  "

                        lines.append(f"{marker} 👤 {msg_username} | 📅 {msg_date}")
                        lines.append(f"   > {msg_text}")

        elif synthesized.supporting_quotes:
            # Fallback if no context provided
            lines.append("")
            lines.append("━" * 30)
            lines.append("📜 Підтверджуючі цитати:")

            for result in synthesized.supporting_quotes[:3]:
                meta = result.get("metadata", {})
                username = meta.get("display_name", "Unknown")
                date = meta.get("formatted_date", "Unknown")
                text = result.get("text", "")

                if len(text) > 800:
                    text = text[:800] + "..."

                lines.append("")
                lines.append(f"💬 {username}:")
                lines.append(f"👤 {username} | 📅 {date}")
                lines.append(f"> {text}")

        lines.append("")
        lines.append("━" * 30)
        lines.append("💡 Відповідай на це повідомлення для уточнення")

        return "\n".join(lines)

    @staticmethod
    def format_synthesized_answer(
        question: str,
        synthesized,
        mentioned_users: list[tuple[int, str]] = None,
        date_from: str = None,
        date_to: str = None,
        sort_order: str = "relevance"
    ) -> str:
        """Format AI-synthesized answer with supporting quotes."""
        header = MessageFormatter.format_synthesized_answer_header(
            question=question,
            date_from=date_from,
            date_to=date_to,
            sort_order=sort_order
        )
        return MessageFormatter.format_synthesized_answer_with_answer(
            header=header,
            answer=synthesized.answer,
            synthesized=synthesized
        )

    # === Analytics Formatting ===

    @staticmethod
    def format_top_speakers(stats: list[dict], limit: int = 10) -> str:
        """
        Format top speakers list.

        Example output:
        🏆 Найактивніші учасники:
        1. 👤 Username1 — 1,234 повідомлень
        2. 👤 Username2 — 987 повідомлень
        """
        if not stats:
            return "Не знайдено повідомлень для аналізу."

        lines = ["🏆 Найактивніші учасники:", ""]
        medals = ["🥇", "🥈", "🥉"]

        for i, stat in enumerate(stats[:limit]):
            medal = medals[i] if i < 3 else f"{i + 1}."
            lines.append(
                f"{medal} 👤 {stat['display_name']} — {stat['message_count']:,} повідомлень"
            )

        return "\n".join(lines)

    @staticmethod
    def format_mention_stats(term: str, stats: list[dict]) -> str:
        """
        Format mention count stats.

        Example output:
        📊 Хто згадував "Зеленський":
        1. 👤 Username1 — 45 разів
        2. 👤 Username2 — 32 рази
        """
        if not stats:
            return f"Ніхто не згадував «{term}»."

        lines = [f"📊 Хто згадував «{term}»:", ""]

        for i, stat in enumerate(stats[:10]):
            rank = i + 1
            count = stat['mention_count']
            suffix = MessageFormatter._pluralize_times(count)
            lines.append(
                f"{rank}. 👤 {stat['display_name']} — {count} {suffix}"
            )

        return "\n".join(lines)

    @staticmethod
    def format_behavioral_stats(trait: str, stats: list[dict]) -> str:
        """
        Format behavioral analysis stats.

        Example output:
        🔍 Хто найбільш "злий":
        1. 👤 Username1 — оцінка 8.5/10 (знайдено 23 повідомлення)
        2. 👤 Username2 — оцінка 7.2/10 (знайдено 15 повідомлень)
        """
        if not stats:
            return f"Не вдалося проаналізувати характеристику «{trait}»."

        # Translate common traits
        trait_names = {
            'angry': 'злий/агресивний',
            'strict': 'строгий',
            'positive': 'позитивний',
            'negative': 'негативний',
            'psycho': 'божевільний',
            'swears': 'лається',
            'kind': 'добрий',
            'aggressive': 'агресивний',
            'toxic': 'токсичний',
        }
        trait_display = trait_names.get(trait, trait)

        lines = [f"🔍 Хто найбільш «{trait_display}»:", ""]

        for i, stat in enumerate(stats[:5]):
            rank = i + 1
            score = stat['score']
            count = stat['example_count']
            lines.append(
                f"{rank}. 👤 {stat['display_name']} — оцінка {score:.1f}/10 "
                f"(знайдено {count} повідомлень)"
            )

        return "\n".join(lines)

    @staticmethod
    def _pluralize_times(count: int) -> str:
        """Pluralize 'раз/рази/разів' in Ukrainian."""
        if count % 10 == 1 and count % 100 != 11:
            return "раз"
        elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
            return "рази"
        else:
            return "разів"

    @staticmethod
    def format_quote_with_forward(result: dict) -> str:
        """Format a search result quote, indicating if it's forwarded."""
        meta = result.get("metadata", {})
        username = meta.get("display_name", "Unknown")
        date = meta.get("formatted_date", "Unknown date")
        text = result.get("text", "")
        is_forwarded = meta.get("is_forwarded", False)
        forward_from = meta.get("forward_from")

        return MessageFormatter.format_quote(
            text=text,
            username=username,
            date=date,
            similarity=result.get("similarity"),
            is_forwarded=is_forwarded,
            forward_from=forward_from
        )
