"""
Telegram bot command handlers.
"""
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

import config
from db import Database
from search import VectorStore, FlipDetector, EmbeddingService
from search.embeddings import ChatService
from search.question_parser import QuestionParser
from search.answer_synthesizer import AnswerSynthesizer
from ingestion import ExportUploader
from .formatters import MessageFormatter
from .conversation_context import ConversationContext

logger = logging.getLogger(__name__)

BOT_USERNAME = "dobby_the_free_trader_bot"


class BotHandlers:
    """Telegram bot command handlers."""

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        flip_detector: FlipDetector,
        uploader: ExportUploader,
        question_parser: QuestionParser,
        answer_synthesizer: AnswerSynthesizer
    ):
        self.db = db
        self.vector_store = vector_store
        self.flip_detector = flip_detector
        self.uploader = uploader
        self.question_parser = question_parser
        self.answer_synthesizer = answer_synthesizer
        self.formatter = MessageFormatter()
        self.conversation_context = ConversationContext()
        # Cache for search queries by message_id (for context highlighting)
        self._search_query_cache: dict[int, str] = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(self.formatter.format_help())

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(self.formatter.format_help())

    async def search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command."""
        if not context.args:
            await update.message.reply_text(
                "❌ Вкажіть запит для пошуку\n"
                "Приклад: /search криптовалюти інвестиції"
            )
            return

        query = " ".join(context.args)
        await update.message.reply_text(f"🔍 Шукаю: \"{query}\"...")

        try:
            results = self.vector_store.search(
                query=query,
                n_results=config.DEFAULT_SEARCH_LIMIT
            )

            response = self.formatter.format_search_results(query, results)
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Search error: {e}")
            await update.message.reply_text(f"❌ Помилка пошуку: {e}")

    async def quote(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /quote @user topic command."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Вкажіть користувача та тему\n"
                "Приклад: /quote @ivan_petrov біткоін"
            )
            return

        user = context.args[0]
        topic = " ".join(context.args[1:])

        await update.message.reply_text(
            f"🔍 Шукаю повідомлення {user} про \"{topic}\"..."
        )

        try:
            results = self.vector_store.search_by_user(
                query=topic,
                user_identifier=user,
                n_results=config.DEFAULT_SEARCH_LIMIT
            )

            if not results:
                await update.message.reply_text(
                    f"❌ Не знайдено повідомлень від {user} про \"{topic}\""
                )
                return

            response = self.formatter.format_search_results(
                f"{user} про {topic}",
                results
            )
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Quote error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def flip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /flip @user topic command."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Вкажіть користувача та тему\n"
                "Приклад: /flip @ivan_petrov крипта"
            )
            return

        user = context.args[0]
        topic = " ".join(context.args[1:])

        await update.message.reply_text(
            f"🔄 Аналізую позицію {user} щодо \"{topic}\"..."
        )

        try:
            result = await self.flip_detector.detect_flip_async(
                user=user,
                topic=topic
            )

            response = self.formatter.format_flip_result(result)
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Flip detection error: {e}")
            await update.message.reply_text(f"❌ Помилка аналізу: {e}")

    async def context_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /context message_id command."""
        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text(
                "❌ Вкажіть ID повідомлення\n"
                "Приклад: /context 12345"
            )
            return

        message_id = int(context.args[0])

        try:
            target = self.db.get_message_by_telegram_id(message_id)
            if not target:
                await update.message.reply_text(
                    f"❌ Повідомлення #{message_id} не знайдено"
                )
                return

            context_msgs = self.db.get_thread_context(message_id)
            response = self.formatter.format_context(target, context_msgs)
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Context error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        try:
            db_stats = self.db.get_stats()
            db_stats["embedded_messages"] = self.vector_store.count()

            response = self.formatter.format_stats(db_stats)
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Stats error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upload command (reply to zip file)."""
        # Check if replying to a document
        reply = update.message.reply_to_message
        if not reply or not reply.document:
            await update.message.reply_text(
                "❌ Відповідайте цією командою на .zip файл\n"
                "1. Надішліть .zip експорт чату\n"
                "2. Відповідайте на нього командою /upload"
            )
            return

        doc = reply.document
        if not doc.file_name.endswith(".zip"):
            await update.message.reply_text("❌ Файл має бути .zip архівом")
            return

        await update.message.reply_text("⏳ Завантажую та перевіряю файл...")

        try:
            # Download file
            file = await context.bot.get_file(doc.file_id)
            zip_path = config.CHAT_EXPORTS_DIR / f"_upload_{doc.file_name}"
            await file.download_to_drive(zip_path)

            # Process upload
            success, message, stats = await self.uploader.process_upload(zip_path)

            if success:
                await update.message.reply_text(
                    self.formatter.format_upload_success(stats)
                )
                # TODO: Trigger indexing
            else:
                await update.message.reply_text(
                    self.formatter.format_upload_error(message)
                )

        except Exception as e:
            logger.error(f"Upload error: {e}")
            await update.message.reply_text(f"❌ Помилка завантаження: {e}")

    async def handle_mention(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle @bot_username mentions with questions."""
        message = update.message
        if not message or not message.text:
            return

        text = message.text
        bot_mention = f"@{BOT_USERNAME}"

        # Check if bot is mentioned
        if bot_mention.lower() not in text.lower():
            return

        # Extract question (remove bot mention)
        question = text.replace(bot_mention, "").replace(f"@{BOT_USERNAME.lower()}", "").strip()
        if not question:
            await message.reply_text("❓ Задай питання після згадки бота")
            return

        await message.reply_text(f"🔍 Аналізую питання...")

        try:
            # Get aliases and users
            aliases_dict = self.db.get_aliases_dict()
            users = self.db.get_all_users()

            # Parse the question with AI
            parsed = await self.question_parser.parse_async(question, aliases_dict, users)

            logger.info(f"Parsed question: {parsed}")

            # Search based on parsed query (get more results for AI analysis)
            if parsed.mentioned_users:
                # Search for specific users
                all_results = []
                for user_id, username in parsed.mentioned_users:
                    results = self.vector_store.search_by_user(
                        query=parsed.search_query,
                        user_identifier=str(user_id),
                        n_results=config.MAX_SEARCH_LIMIT,
                        date_from=parsed.date_from,
                        date_to=parsed.date_to
                    )
                    all_results.extend(results)
            else:
                # General search
                all_results = self.vector_store.search(
                    query=parsed.search_query,
                    n_results=config.MAX_SEARCH_LIMIT,
                    date_from=parsed.date_from,
                    date_to=parsed.date_to
                )

            # Apply sorting if requested
            if parsed.sort_order != "relevance":
                all_results = self._sort_results(all_results, parsed.sort_order)

            # Synthesize answer using AI
            synthesized = await self.answer_synthesizer.synthesize_async(
                question=parsed.original_question,
                results=all_results,
                mentioned_users=parsed.mentioned_users
            )

            # Format and send response
            response = self.formatter.format_synthesized_answer(
                question=parsed.original_question,
                synthesized=synthesized,
                mentioned_users=parsed.mentioned_users,
                date_from=parsed.date_from,
                date_to=parsed.date_to,
                sort_order=parsed.sort_order
            )

            # Build context buttons for supporting quotes
            keyboard = self._build_context_keyboard(synthesized.supporting_quotes)

            # Cache search query for each shown message (for context highlighting)
            for quote in synthesized.supporting_quotes[:3]:
                msg_id = quote.get("metadata", {}).get("message_id")
                if msg_id:
                    self._search_query_cache[msg_id] = parsed.search_query

            sent_message = await message.reply_text(
                response,
                reply_markup=keyboard
            )

            # Store context for follow-ups
            self.conversation_context.store(
                chat_id=message.chat_id,
                bot_message_id=sent_message.message_id,
                original_question=parsed.original_question,
                search_query=parsed.search_query,
                mentioned_users=parsed.mentioned_users,
                results=all_results
            )

        except Exception as e:
            logger.error(f"Mention handler error: {e}", exc_info=True)
            await message.reply_text(f"❌ Помилка обробки питання: {e}")

    def _sort_results(self, results: list[dict], sort_order: str) -> list[dict]:
        """Sort results by date."""
        def get_timestamp(result):
            meta = result.get("metadata", {})
            ts = meta.get("timestamp_unix")
            if ts:
                return int(ts)
            # Fallback: parse formatted_date
            formatted = meta.get("formatted_date", "")
            if formatted:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(formatted.split()[0], "%d.%m.%Y")
                    return int(dt.timestamp())
                except ValueError:
                    pass
            return 0

        reverse = (sort_order == "newest")
        return sorted(results, key=get_timestamp, reverse=reverse)

    def _build_context_keyboard(self, quotes: list[dict]) -> InlineKeyboardMarkup:
        """Build inline keyboard with context buttons for quotes."""
        if not quotes:
            return None

        buttons = []
        for i, quote in enumerate(quotes[:3], 1):
            meta = quote.get("metadata", {})
            msg_id = meta.get("message_id")
            if msg_id:
                buttons.append(
                    InlineKeyboardButton(
                        text=f"📜 Контекст [{i}]",
                        callback_data=f"ctx:{msg_id}"
                    )
                )

        if not buttons:
            return None

        # Arrange in a row
        return InlineKeyboardMarkup([buttons])

    async def handle_context_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle context button clicks."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if not data.startswith("ctx:"):
            return

        try:
            message_id = int(data.split(":")[1])

            target = self.db.get_message_by_telegram_id(message_id)
            if not target:
                await query.message.reply_text(f"❌ Повідомлення #{message_id} не знайдено")
                return

            context_msgs = self.db.get_thread_context(message_id)

            # Get cached search query for highlighting
            search_query = self._search_query_cache.get(message_id)

            response = self.formatter.format_context(
                target,
                context_msgs,
                search_query=search_query
            )
            await query.message.reply_text(response)

        except Exception as e:
            logger.error(f"Context callback error: {e}")
            await query.message.reply_text(f"❌ Помилка: {e}")

    async def handle_followup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle follow-up questions (replies to bot messages)."""
        message = update.message
        if not message or not message.reply_to_message:
            return

        reply_to = message.reply_to_message
        chat_id = message.chat_id

        # Get previous context
        prev_context = self.conversation_context.get(chat_id, reply_to.message_id)
        if not prev_context:
            return  # Not a reply to our message

        question = message.text
        if not question:
            return

        await message.reply_text("🔍 Аналізую уточнення...")

        try:
            # Get aliases and users
            aliases_dict = self.db.get_aliases_dict()
            users = self.db.get_all_users()

            # Parse the follow-up with context
            parsed = await self.question_parser.parse_async(question, aliases_dict, users)

            # Use previous users if none mentioned
            mentioned_users = parsed.mentioned_users or prev_context.mentioned_users

            # Check for "more" or "show more" patterns
            more_patterns = ["ще", "більше", "more", "покажи ще", "ещё"]
            is_more_request = any(p in question.lower() for p in more_patterns)

            if is_more_request:
                # Get more results with same query
                if mentioned_users:
                    all_results = []
                    for user_id, username in mentioned_users:
                        results = self.vector_store.search_by_user(
                            query=prev_context.search_query,
                            user_identifier=str(user_id),
                            n_results=config.MAX_SEARCH_LIMIT
                        )
                        all_results.extend(results)
                else:
                    all_results = self.vector_store.search(
                        query=prev_context.search_query,
                        n_results=config.MAX_SEARCH_LIMIT
                    )

                # Skip already shown results
                shown_ids = {r.get("id") for r in prev_context.results}
                new_results = [r for r in all_results if r.get("id") not in shown_ids][:10]
            else:
                # New refined search
                if mentioned_users:
                    all_results = []
                    for user_id, username in mentioned_users:
                        results = self.vector_store.search_by_user(
                            query=parsed.search_query,
                            user_identifier=str(user_id),
                            n_results=config.DEFAULT_SEARCH_LIMIT
                        )
                        all_results.extend(results)
                else:
                    all_results = self.vector_store.search(
                        query=parsed.search_query,
                        n_results=config.DEFAULT_SEARCH_LIMIT
                    )
                new_results = all_results

            # Format and send
            response = self.formatter.format_ai_response(
                question=question,
                results=new_results,
                mentioned_users=mentioned_users
            )

            sent_message = await message.reply_text(response)

            # Store new context
            self.conversation_context.store(
                chat_id=chat_id,
                bot_message_id=sent_message.message_id,
                original_question=question,
                search_query=parsed.search_query,
                mentioned_users=mentioned_users,
                results=new_results
            )

        except Exception as e:
            logger.error(f"Follow-up handler error: {e}", exc_info=True)
            await message.reply_text(f"❌ Помилка: {e}")

    async def aliases(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /aliases command - list all aliases."""
        try:
            aliases = self.db.get_all_aliases()
            users = self.db.get_all_users()
            response = self.formatter.format_aliases(aliases, users)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Aliases error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def add_alias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alias command - add alias for user."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Вкажіть користувача та прізвисько\n"
                "Приклад: /alias Женек бушуев"
            )
            return

        username = context.args[0].lstrip("@")
        alias = " ".join(context.args[1:])

        try:
            # Find user by username
            users = self.db.get_all_users()
            user_match = None
            for user_id, uname in users:
                if uname and uname.lower() == username.lower():
                    user_match = (user_id, uname)
                    break

            if not user_match:
                # Try partial match
                for user_id, uname in users:
                    if uname and username.lower() in uname.lower():
                        user_match = (user_id, uname)
                        break

            if not user_match:
                await update.message.reply_text(
                    f"❌ Користувача '{username}' не знайдено\n"
                    f"Доступні: {', '.join(u[1] for u in users if u[1])}"
                )
                return

            user_id, actual_username = user_match
            result = self.db.add_alias(user_id, actual_username, alias)

            if result:
                await update.message.reply_text(
                    f"✅ Додано прізвисько '{alias}' для {actual_username}"
                )
            else:
                await update.message.reply_text(
                    f"❌ Прізвисько '{alias}' вже існує"
                )

        except Exception as e:
            logger.error(f"Add alias error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def remove_alias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alias_remove command."""
        if not context.args:
            await update.message.reply_text(
                "❌ Вкажіть прізвисько для видалення\n"
                "Приклад: /alias_remove бушуев"
            )
            return

        alias = " ".join(context.args)

        try:
            if self.db.remove_alias(alias):
                await update.message.reply_text(f"✅ Прізвисько '{alias}' видалено")
            else:
                await update.message.reply_text(f"❌ Прізвисько '{alias}' не знайдено")
        except Exception as e:
            logger.error(f"Remove alias error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def seed_aliases(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /seed_aliases command - add predefined aliases."""
        PREDEFINED_ALIASES = {
            369544572: {
                "username": "Женек",
                "aliases": ["евгеша", "жидос", "бушуєв", "бушуев", "буш", "жень-мень", "жменька"]
            },
            325310655: {
                "username": "S D",
                "aliases": ["серёга", "серж", "сержик", "дрож", "сион"]
            },
            645706876: {
                "username": "Лех Качинський",
                "aliases": ["гусь", "птица", "качка", "качур", "гусенко", "проценко", "птиценко"]
            }
        }

        added = 0
        skipped = 0

        for user_id, info in PREDEFINED_ALIASES.items():
            for alias in info["aliases"]:
                result = self.db.add_alias(user_id, info["username"], alias)
                if result:
                    added += 1
                else:
                    skipped += 1

        await update.message.reply_text(
            f"✅ Прізвиська налаштовані\n"
            f"Додано: {added}\n"
            f"Пропущено (вже існують): {skipped}"
        )


def setup_handlers(app: Application, db: Database, vector_store: VectorStore):
    """Set up all bot handlers."""
    embedding_service = EmbeddingService()
    chat_service = ChatService()
    flip_detector = FlipDetector(vector_store, chat_service)
    uploader = ExportUploader(db)
    question_parser = QuestionParser()
    answer_synthesizer = AnswerSynthesizer()

    handlers = BotHandlers(
        db=db,
        vector_store=vector_store,
        flip_detector=flip_detector,
        uploader=uploader,
        question_parser=question_parser,
        answer_synthesizer=answer_synthesizer
    )

    # Command handlers (minimal - AI handles most queries via @mention)
    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(CommandHandler("help", handlers.help_command))
    app.add_handler(CommandHandler("context", handlers.context_command))
    app.add_handler(CommandHandler("stats", handlers.stats))
    app.add_handler(CommandHandler("upload", handlers.upload))

    # Alias management
    app.add_handler(CommandHandler("aliases", handlers.aliases))
    app.add_handler(CommandHandler("alias", handlers.add_alias))
    app.add_handler(CommandHandler("alias_remove", handlers.remove_alias))
    app.add_handler(CommandHandler("seed_aliases", handlers.seed_aliases))

    # Mention handler (for @bot questions)
    mention_filter = filters.TEXT & filters.Regex(f"(?i)@{BOT_USERNAME}")
    app.add_handler(MessageHandler(mention_filter, handlers.handle_mention))

    # Follow-up handler (replies to bot messages)
    reply_filter = filters.TEXT & filters.REPLY
    app.add_handler(MessageHandler(reply_filter, handlers.handle_followup))

    # Callback handler for inline buttons (context buttons)
    app.add_handler(CallbackQueryHandler(handlers.handle_context_callback, pattern=r"^ctx:"))

    return handlers
