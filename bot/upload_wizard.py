"""
Upload wizard for guided chat export uploading.
"""
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class UploadStep(Enum):
    """Upload wizard steps."""
    INSTRUCTIONS = 1
    WAITING_FILE = 2
    PROCESSING = 3
    INDEXING = 4
    COMPLETE = 5
    ERROR = 6


@dataclass
class UploadSession:
    """Track upload wizard session state."""
    chat_id: int
    user_id: int
    step: UploadStep = UploadStep.INSTRUCTIONS
    file_path: Optional[Path] = None
    chat_name: Optional[str] = None
    message_count: int = 0
    indexed_count: int = 0
    error_message: Optional[str] = None
    status_message_id: Optional[int] = None


class UploadWizard:
    """
    Manages upload wizard sessions for guided chat export uploading.
    """

    def __init__(self):
        # Map: (chat_id, user_id) -> UploadSession
        self._sessions: dict[tuple[int, int], UploadSession] = {}

    def start_session(self, chat_id: int, user_id: int) -> UploadSession:
        """Start a new upload session."""
        key = (chat_id, user_id)
        session = UploadSession(chat_id=chat_id, user_id=user_id)
        self._sessions[key] = session
        return session

    def get_session(self, chat_id: int, user_id: int) -> Optional[UploadSession]:
        """Get existing session."""
        return self._sessions.get((chat_id, user_id))

    def update_session(self, session: UploadSession) -> None:
        """Update session state."""
        key = (session.chat_id, session.user_id)
        self._sessions[key] = session

    def end_session(self, chat_id: int, user_id: int) -> None:
        """End and remove session."""
        key = (chat_id, user_id)
        if key in self._sessions:
            del self._sessions[key]

    def is_waiting_for_file(self, chat_id: int, user_id: int) -> bool:
        """Check if session is waiting for file upload."""
        session = self.get_session(chat_id, user_id)
        return session is not None and session.step == UploadStep.WAITING_FILE

    @staticmethod
    def format_instructions() -> str:
        """Format upload instructions message."""
        return """📤 Завантаження експорту чату

Для завантаження історії чату виконай наступні кроки:

1️⃣ Відкрий Telegram Desktop
2️⃣ Перейди в потрібний чат
3️⃣ Натисни ⋮ → "Експортувати історію чату"
4️⃣ Вибери формат JSON
5️⃣ Зачекай завершення експорту
6️⃣ Надішли отриманий .zip файл сюди

⚠️ ВАЖЛИВО:
• Використовуй ТІЛЬКИ Telegram Desktop
• Формат має бути JSON (не HTML)
• Максимальний розмір: 50MB

📎 Надішли .zip файл коли будеш готовий
Або натисни "Скасувати" для виходу"""

    @staticmethod
    def format_processing(filename: str, progress: int = 0) -> str:
        """Format processing status message."""
        bar_length = 20
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        return f"""📦 Обробка файлу: {filename}

{bar} {progress}%

⏳ Розпаковую та перевіряю структуру..."""

    @staticmethod
    def format_indexing(processed: int, total: int) -> str:
        """Format indexing progress message."""
        progress = int((processed / total) * 100) if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        return f"""🔍 Індексація повідомлень

{bar} {progress}%

📨 Оброблено: {processed:,} / {total:,} повідомлень

⏳ Це може зайняти декілька хвилин..."""

    @staticmethod
    def format_complete(chat_name: str, message_count: int, indexed_count: int) -> str:
        """Format completion message."""
        return f"""✅ Експорт успішно завантажено!

💬 Чат: {chat_name}
📨 Повідомлень: {message_count:,}
🔍 Проіндексовано: {indexed_count:,}

Тепер можеш шукати в історії чату:
@dobby_the_free_trader_bot що казав хтось про щось?

📋 Доступні команди:
/stats — Статистика бази
/mystats — Статистика користувачів
/aliases — Список прізвиськ"""

    @staticmethod
    def format_error(error: str) -> str:
        """Format error message."""
        return f"""❌ Помилка завантаження

{error}

Спробуй ще раз або перевір формат файлу.
Натисни /upload щоб почати знову."""


# Keyboard builders for wizard
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def build_instructions_keyboard() -> InlineKeyboardMarkup:
    """Build keyboard for instructions step."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                text="❓ Детальна інструкція",
                callback_data="upload:help"
            )
        ],
        [
            InlineKeyboardButton(
                text="❌ Скасувати",
                callback_data="upload:cancel"
            )
        ]
    ])


def build_processing_keyboard() -> InlineKeyboardMarkup:
    """Build keyboard for processing step."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                text="❌ Скасувати",
                callback_data="upload:cancel"
            )
        ]
    ])


def build_complete_keyboard() -> InlineKeyboardMarkup:
    """Build keyboard for completion step."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                text="📤 Завантажити ще",
                callback_data="upload:restart"
            ),
            InlineKeyboardButton(
                text="📊 Статистика",
                callback_data="upload:stats"
            )
        ]
    ])


def build_error_keyboard() -> InlineKeyboardMarkup:
    """Build keyboard for error step."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                text="🔄 Спробувати ще",
                callback_data="upload:restart"
            )
        ],
        [
            InlineKeyboardButton(
                text="❌ Закрити",
                callback_data="upload:cancel"
            )
        ]
    ])


DETAILED_HELP = """📖 Детальна інструкція з експорту

🖥️ Telegram Desktop (Windows/Mac/Linux):

1. Відкрий програму Telegram Desktop
2. Перейди в чат, який хочеш експортувати
3. Натисни на три крапки ⋮ у верхньому правому куті
4. Вибери "Експортувати історію чату"
5. У вікні налаштувань:
   • Формат: JSON
   • Медіа: можна не включати (економить місце)
   • Період: весь час
6. Натисни "Експортувати"
7. Зачекай завершення (може зайняти час)
8. Знайди папку з результатом
9. Заархівуй її в .zip файл
10. Надішли .zip сюди

⚠️ Поширені помилки:

• "Неправильний формат" - переконайся що вибрав JSON
• "Файл занадто великий" - розбий експорт на частини
• "Не знайдено повідомлень" - перевір чи є result.json

📱 Через мобільний додаток експорт НЕ працює!"""
