"""
Telegram bot command handlers.
"""
import logging
import time
from datetime import datetime, timedelta
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
from telegram.constants import ChatAction

import config
from db import Database
from search import VectorStore, FlipDetector, EmbeddingService
from search.embeddings import ChatService
from search.question_parser import QuestionParser
from search.answer_synthesizer import AnswerSynthesizer
from search.search_agent import SearchAgent
from search.entity_aliases import (
    reload_aliases as reload_entity_aliases,
    ENTITY_ALIASES,
    HARDCODED_ALIASES
)
from ingestion import ExportUploader
from .formatters import MessageFormatter
from .conversation_context import ConversationContext
from .upload_wizard import (
    UploadWizard,
    UploadStep,
    build_instructions_keyboard,
    build_processing_keyboard,
    build_complete_keyboard,
    build_error_keyboard,
    DETAILED_HELP
)

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
        self.search_agent = SearchAgent(vector_store)
        self.formatter = MessageFormatter()
        self.conversation_context = ConversationContext()
        self.upload_wizard = UploadWizard()
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

            # Get related context (filtered by topic relevance)
            context_msgs = await self._get_related_context(target, message_id)
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

    async def mystats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mystats command - show user statistics dashboard."""
        try:
            await update.message.reply_text("📊 Збираю статистику...")

            # Get user statistics
            user_stats = self.db.get_user_stats(limit=10)
            hourly_distribution = self.db.get_hourly_distribution()
            db_stats = self.db.get_stats()

            response = self.formatter.format_user_stats(
                user_stats=user_stats,
                hourly_distribution=hourly_distribution,
                total_messages=db_stats.get("total_messages", 0)
            )
            await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"User stats error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /upload command - start upload wizard."""
        message = update.message
        chat_id = message.chat_id
        user_id = message.from_user.id

        # Check if replying to a document (old behavior for backwards compatibility)
        reply = message.reply_to_message
        if reply and reply.document and reply.document.file_name.endswith(".zip"):
            # Direct upload via reply
            await self._process_upload_file(update, context, reply.document)
            return

        # Start wizard
        session = self.upload_wizard.start_session(chat_id, user_id)
        session.step = UploadStep.WAITING_FILE

        instructions = UploadWizard.format_instructions()
        keyboard = build_instructions_keyboard()

        sent = await message.reply_text(instructions, reply_markup=keyboard)
        session.status_message_id = sent.message_id
        self.upload_wizard.update_session(session)

    async def handle_document_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle ZIP file uploads during wizard."""
        message = update.message
        if not message or not message.document:
            return

        doc = message.document
        chat_id = message.chat_id
        user_id = message.from_user.id

        # Check if user has active wizard session
        if not self.upload_wizard.is_waiting_for_file(chat_id, user_id):
            # No active session - could be regular file share
            return

        # Validate file
        if not doc.file_name or not doc.file_name.endswith(".zip"):
            await message.reply_text("❌ Файл має бути .zip архівом")
            return

        await self._process_upload_file(update, context, doc)

    async def _process_upload_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE, doc):
        """Process uploaded ZIP file."""
        message = update.message
        chat_id = message.chat_id
        user_id = message.from_user.id

        # Get or create session
        session = self.upload_wizard.get_session(chat_id, user_id)
        if not session:
            session = self.upload_wizard.start_session(chat_id, user_id)

        session.step = UploadStep.PROCESSING

        # Send processing status
        status_msg = await message.reply_text(
            UploadWizard.format_processing(doc.file_name, 10),
            reply_markup=build_processing_keyboard()
        )
        session.status_message_id = status_msg.message_id
        self.upload_wizard.update_session(session)

        try:
            # Download file
            await status_msg.edit_text(
                UploadWizard.format_processing(doc.file_name, 30)
            )

            file = await context.bot.get_file(doc.file_id)
            zip_path = config.CHAT_EXPORTS_DIR / f"_upload_{doc.file_name}"
            await file.download_to_drive(zip_path)
            session.file_path = zip_path

            await status_msg.edit_text(
                UploadWizard.format_processing(doc.file_name, 60)
            )

            # Process upload
            success, error_msg, stats = await self.uploader.process_upload(zip_path)

            if not success:
                session.step = UploadStep.ERROR
                session.error_message = error_msg
                self.upload_wizard.update_session(session)
                await status_msg.edit_text(
                    UploadWizard.format_error(error_msg),
                    reply_markup=build_error_keyboard()
                )
                return

            session.chat_name = stats.get("chat_name", "Unknown")
            session.message_count = stats.get("estimated_messages", 0)

            # Update to indexing status
            session.step = UploadStep.INDEXING
            await status_msg.edit_text(
                UploadWizard.format_indexing(0, session.message_count)
            )

            # Note: Actual indexing would happen here via uploader
            # For now, simulate progress
            session.indexed_count = session.message_count

            # Complete
            session.step = UploadStep.COMPLETE
            self.upload_wizard.update_session(session)

            await status_msg.edit_text(
                UploadWizard.format_complete(
                    chat_name=session.chat_name,
                    message_count=session.message_count,
                    indexed_count=session.indexed_count
                ),
                reply_markup=build_complete_keyboard()
            )

            # End session
            self.upload_wizard.end_session(chat_id, user_id)

        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            session.step = UploadStep.ERROR
            session.error_message = str(e)
            self.upload_wizard.update_session(session)

            await status_msg.edit_text(
                UploadWizard.format_error(str(e)),
                reply_markup=build_error_keyboard()
            )

    async def handle_upload_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle upload wizard callback buttons."""
        query = update.callback_query

        data = query.data
        if not data.startswith("upload:"):
            return

        action = data.split(":")[1]
        chat_id = query.message.chat_id
        user_id = query.from_user.id

        try:
            if action == "cancel":
                self.upload_wizard.end_session(chat_id, user_id)
                await query.message.edit_text("❌ Завантаження скасовано")
                await query.answer()

            elif action == "restart":
                # Start new session
                session = self.upload_wizard.start_session(chat_id, user_id)
                session.step = UploadStep.WAITING_FILE
                self.upload_wizard.update_session(session)

                await query.message.edit_text(
                    UploadWizard.format_instructions(),
                    reply_markup=build_instructions_keyboard()
                )
                await query.answer()

            elif action == "help":
                await query.answer()
                await query.message.reply_text(DETAILED_HELP)

            elif action == "stats":
                await query.answer()
                # Trigger stats command
                db_stats = self.db.get_stats()
                db_stats["embedded_messages"] = self.vector_store.count()
                response = self.formatter.format_stats(db_stats)
                await query.message.reply_text(response)

        except Exception as e:
            logger.error(f"Upload callback error: {e}")
            await query.answer(f"❌ Помилка: {e}", show_alert=True)

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

        # Send status message that we'll edit later
        status_msg = await message.reply_text("🔍 Аналізую питання...")

        try:
            # Get aliases and users
            aliases_dict = self.db.get_aliases_dict()
            users = self.db.get_all_users()

            # Parse the question with AI
            parsed = await self.question_parser.parse_async(question, aliases_dict, users)

            logger.info(f"Parsed question: {parsed}")

            # Multi-step search with automatic query reformulation
            if parsed.mentioned_users:
                # Search for specific users with agent
                all_results = []
                for user_id, username in parsed.mentioned_users:
                    results, session = await self.search_agent.search(
                        question=parsed.original_question,
                        initial_query=parsed.search_query,
                        user_id=user_id,
                        date_from=parsed.date_from,
                        date_to=parsed.date_to,
                        min_relevant=3,
                        max_iterations=3
                    )
                    all_results.extend(results)
                    logger.info(f"Search for user {username}: {len(results)} results after {session.iterations} iterations")
            else:
                # General search with agent
                all_results, session = await self.search_agent.search(
                    question=parsed.original_question,
                    initial_query=parsed.search_query,
                    date_from=parsed.date_from,
                    date_to=parsed.date_to,
                    min_relevant=3,
                    max_iterations=3
                )
                logger.info(f"Search: {len(all_results)} results after {session.iterations} iterations, queries: {session.queries_tried}")

            # Apply sorting if requested
            if parsed.sort_order != "relevance":
                all_results = self._sort_results(all_results, parsed.sort_order)

            # Send typing indicator
            await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)

            # Update status to show we're synthesizing
            await status_msg.edit_text("🤖 Генерую відповідь...")

            # Build header for streamed response
            header = self.formatter.format_synthesized_answer_header(
                question=parsed.original_question,
                date_from=parsed.date_from,
                date_to=parsed.date_to,
                sort_order=parsed.sort_order
            )

            # Synthesize answer with streaming
            stream = self.answer_synthesizer.synthesize_stream_async(
                question=parsed.original_question,
                results=all_results,
                mentioned_users=parsed.mentioned_users,
                filter_relevance=False  # Already filtered by agent
            )

            answer_text, synthesized = await self._stream_response(
                status_msg=status_msg,
                stream=stream,
                header=header,
                update_interval=1.5
            )

            # Format final response with quotes
            response = self.formatter.format_synthesized_answer_with_answer(
                header=header,
                answer=answer_text,
                synthesized=synthesized
            )

            # Cache search query for each shown message (for context highlighting)
            for quote in synthesized.supporting_quotes[:3]:
                msg_id = quote.get("metadata", {}).get("message_id")
                if msg_id:
                    self._search_query_cache[msg_id] = parsed.search_query

            # Edit status message with final response (or send new if too long/edit fails)
            try:
                # First edit without keyboard to get message_id
                if len(response) <= 4096:
                    await status_msg.edit_text(response)
                    sent_message = status_msg
                else:
                    truncated = response[:4000] + "\n\n⚠️ Відповідь скорочено..."
                    await status_msg.edit_text(truncated)
                    sent_message = status_msg
            except Exception as edit_error:
                logger.warning(f"Failed to edit status message: {edit_error}")
                await status_msg.delete()
                sent_message = await message.reply_text(response)

            # Store context for follow-ups with pagination support
            context_key = self.conversation_context.make_context_key(
                message.chat_id, sent_message.message_id
            )
            state = self.conversation_context.store(
                chat_id=message.chat_id,
                bot_message_id=sent_message.message_id,
                original_question=parsed.original_question,
                search_query=parsed.search_query,
                mentioned_users=parsed.mentioned_users,
                results=synthesized.supporting_quotes[:5],  # Page 1 results
                all_results=all_results
            )

            # Generate follow-up suggestions
            try:
                suggestions = await self.question_parser.generate_suggestions_async(
                    question=parsed.original_question,
                    results=all_results[:5]
                )
                for suggestion in suggestions:
                    state.add_suggestion(suggestion)
                self.conversation_context.update_state(state)
            except Exception as e:
                logger.debug(f"Failed to generate suggestions: {e}")
                suggestions = []

            # Build keyboard with context buttons, pagination, and suggestions
            keyboard = self._build_full_keyboard(
                quotes=synthesized.supporting_quotes,
                context_key=context_key,
                current_page=1,
                total_pages=state.total_pages,
                show_pagination=len(all_results) > 5,
                suggestions=suggestions,
                state=state
            )

            # Update message with keyboard
            if keyboard:
                try:
                    final_response = response if len(response) <= 4096 else response[:4000] + "\n\n⚠️ Відповідь скорочено..."
                    await sent_message.edit_text(final_response, reply_markup=keyboard)
                except Exception:
                    pass  # Keyboard update failed, but message is still visible

        except Exception as e:
            logger.error(f"Mention handler error: {e}", exc_info=True)
            # Try to edit status message with error, or send new
            try:
                await status_msg.edit_text(f"❌ Помилка обробки питання: {e}")
            except Exception:
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

    def _build_context_keyboard(
        self,
        quotes: list[dict],
        context_key: str = None,
        current_page: int = 1,
        total_pages: int = 1,
        show_pagination: bool = True,
        show_thread_button: bool = True
    ) -> InlineKeyboardMarkup:
        """Build inline keyboard with context buttons and pagination."""
        rows = []

        # Pagination row (if multiple pages)
        if show_pagination and total_pages > 1 and context_key:
            pagination_buttons = []
            if current_page > 1:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="← Попередні",
                        callback_data=f"page:prev:{context_key}"
                    )
                )
            pagination_buttons.append(
                InlineKeyboardButton(
                    text=f"{current_page}/{total_pages}",
                    callback_data="page:noop"
                )
            )
            if current_page < total_pages:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="Наступні →",
                        callback_data=f"page:next:{context_key}"
                    )
                )
            rows.append(pagination_buttons)

        # Context buttons for quotes
        if quotes:
            context_buttons = []
            for i, quote in enumerate(quotes[:3], 1):
                meta = quote.get("metadata", {})
                msg_id = meta.get("message_id")
                if msg_id:
                    context_buttons.append(
                        InlineKeyboardButton(
                            text=f"📜 Контекст [{i}]",
                            callback_data=f"ctx:{msg_id}"
                        )
                    )
            if context_buttons:
                rows.append(context_buttons)

            # Thread button (show for first quote if it has reply chain)
            if show_thread_button and quotes:
                first_msg_id = quotes[0].get("metadata", {}).get("message_id")
                if first_msg_id:
                    rows.append([
                        InlineKeyboardButton(
                            text="🧵 Показати розмову",
                            callback_data=f"thread:{first_msg_id}"
                        )
                    ])

        if not rows:
            return None

        return InlineKeyboardMarkup(rows)

    async def _stream_response(
        self,
        status_msg,
        stream,
        header: str = "",
        update_interval: float = 2.0
    ) -> str:
        """
        Stream response with progressive message updates.

        Args:
            status_msg: Telegram message to edit
            stream: Async generator yielding (chunk_type, content)
            header: Optional header to prepend to content
            update_interval: Seconds between message edits

        Returns:
            Full response text
        """
        buffer = header
        last_update = time.time()
        final_result = None

        async for chunk_type, content in stream:
            if chunk_type == "answer_chunk":
                buffer += content
                # Update message periodically to show progress
                now = time.time()
                if now - last_update >= update_interval:
                    try:
                        display_text = buffer + "▌"
                        if len(display_text) <= 4096:
                            await status_msg.edit_text(display_text)
                        last_update = now
                    except Exception as e:
                        logger.debug(f"Stream update edit failed: {e}")
            elif chunk_type == "done":
                final_result = content

        return buffer, final_result

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

            # Get related context (filtered by topic relevance)
            context_msgs = await self._get_related_context(target, message_id)

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

    async def handle_pagination_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle pagination button clicks."""
        query = update.callback_query

        data = query.data
        if not data.startswith("page:"):
            return

        try:
            parts = data.split(":")
            action = parts[1]

            if action == "noop":
                await query.answer()
                return

            context_key = parts[2]
            state = self.conversation_context.get_by_context_key(context_key)

            if not state:
                await query.answer("⚠️ Контекст сесії застарів", show_alert=True)
                return

            # Update page
            if action == "prev" and state.current_page > 1:
                state.current_page -= 1
            elif action == "next" and state.current_page < state.total_pages:
                state.current_page += 1
            else:
                await query.answer()
                return

            # Update state
            self.conversation_context.update_state(state)

            # Get results for current page
            page_results = state.get_page_results()

            # Format response with updated page
            response = self.formatter.format_paginated_results(
                question=state.original_question,
                results=page_results,
                current_page=state.current_page,
                total_pages=state.total_pages,
                total_results=len(state.all_results)
            )

            # Build keyboard with updated pagination
            keyboard = self._build_context_keyboard(
                quotes=page_results,
                context_key=context_key,
                current_page=state.current_page,
                total_pages=state.total_pages
            )

            await query.message.edit_text(response, reply_markup=keyboard)
            await query.answer()

        except Exception as e:
            logger.error(f"Pagination callback error: {e}")
            await query.answer(f"❌ Помилка: {e}", show_alert=True)

    def _calculate_date_range(self, filter_type: str) -> tuple[str, str]:
        """Calculate date range for a time filter."""
        today = datetime.now()

        if filter_type == "today":
            date_str = today.strftime("%Y-%m-%d")
            return (date_str, date_str)
        elif filter_type == "week":
            start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            return (start, today.strftime("%Y-%m-%d"))
        elif filter_type == "month":
            start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            return (start, today.strftime("%Y-%m-%d"))
        elif filter_type.isdigit() and len(filter_type) == 4:
            # Year filter
            year = filter_type
            return (f"{year}-01-01", f"{year}-12-31")
        elif filter_type == "reset":
            return (None, None)
        else:
            return (None, None)

    def _filter_results_by_date(
        self,
        results: list[dict],
        date_from: str = None,
        date_to: str = None
    ) -> list[dict]:
        """Filter results by date range."""
        if not date_from and not date_to:
            return results

        filtered = []
        for result in results:
            meta = result.get("metadata", {})
            ts_unix = meta.get("timestamp_unix")
            formatted_date = meta.get("formatted_date", "")

            # Try to get timestamp
            msg_date = None
            if ts_unix:
                msg_date = datetime.fromtimestamp(int(ts_unix)).strftime("%Y-%m-%d")
            elif formatted_date:
                try:
                    # Parse "DD.MM.YYYY HH:MM" format
                    dt = datetime.strptime(formatted_date.split()[0], "%d.%m.%Y")
                    msg_date = dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue

            if msg_date:
                if date_from and msg_date < date_from:
                    continue
                if date_to and msg_date > date_to:
                    continue
                filtered.append(result)

        return filtered

    async def handle_time_filter_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle time filter button clicks."""
        query = update.callback_query

        data = query.data
        if not data.startswith("time:"):
            return

        try:
            parts = data.split(":")
            filter_type = parts[1]
            context_key = parts[2]

            state = self.conversation_context.get_by_context_key(context_key)
            if not state:
                await query.answer("⚠️ Контекст сесії застарів", show_alert=True)
                return

            # Calculate date range
            date_from, date_to = self._calculate_date_range(filter_type)

            # Filter all results
            if filter_type == "reset":
                filtered_results = state.all_results
                state.active_date_filter = None
            else:
                filtered_results = self._filter_results_by_date(
                    state.all_results, date_from, date_to
                )
                state.active_date_filter = filter_type

            # Update state
            state.results = filtered_results
            state.current_page = 1
            self.conversation_context.update_state(state)

            if not filtered_results:
                await query.answer(f"⚠️ Немає повідомлень за цей період", show_alert=True)
                return

            # Format response
            filter_label = self._get_filter_label(filter_type)
            response = self.formatter.format_filtered_results(
                question=state.original_question,
                results=filtered_results[:5],
                filter_label=filter_label,
                total_results=len(filtered_results)
            )

            # Build keyboard with time filters and pagination
            keyboard = self._build_time_filter_keyboard(
                context_key=context_key,
                active_filter=state.active_date_filter,
                current_page=1,
                total_pages=state.total_pages,
                quotes=filtered_results[:3]
            )

            await query.message.edit_text(response, reply_markup=keyboard)
            await query.answer()

        except Exception as e:
            logger.error(f"Time filter callback error: {e}")
            await query.answer(f"❌ Помилка: {e}", show_alert=True)

    def _get_filter_label(self, filter_type: str) -> str:
        """Get human-readable label for filter type."""
        labels = {
            "today": "Сьогодні",
            "week": "Тиждень",
            "month": "Місяць",
            "reset": None
        }
        if filter_type in labels:
            return labels[filter_type]
        if filter_type.isdigit():
            return f"Рік {filter_type}"
        return None

    def _build_time_filter_keyboard(
        self,
        context_key: str,
        active_filter: str = None,
        current_page: int = 1,
        total_pages: int = 1,
        quotes: list[dict] = None
    ) -> InlineKeyboardMarkup:
        """Build keyboard with time filter buttons."""
        rows = []

        # Quick time filters row 1
        time_buttons_1 = []
        for filter_type, label in [("today", "Сьогодні"), ("week", "Тиждень"), ("month", "Місяць")]:
            prefix = "✓ " if active_filter == filter_type else ""
            time_buttons_1.append(
                InlineKeyboardButton(
                    text=f"{prefix}{label}",
                    callback_data=f"time:{filter_type}:{context_key}"
                )
            )
        rows.append(time_buttons_1)

        # Year filters row 2
        current_year = datetime.now().year
        time_buttons_2 = []
        for year in range(current_year, current_year - 3, -1):
            year_str = str(year)
            prefix = "✓ " if active_filter == year_str else ""
            time_buttons_2.append(
                InlineKeyboardButton(
                    text=f"{prefix}{year}",
                    callback_data=f"time:{year_str}:{context_key}"
                )
            )
        rows.append(time_buttons_2)

        # Reset filter button
        if active_filter:
            rows.append([
                InlineKeyboardButton(
                    text="🔄 Скинути фільтр",
                    callback_data=f"time:reset:{context_key}"
                )
            ])

        # Pagination row (if multiple pages)
        if total_pages > 1:
            pagination_buttons = []
            if current_page > 1:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="← Попередні",
                        callback_data=f"page:prev:{context_key}"
                    )
                )
            pagination_buttons.append(
                InlineKeyboardButton(
                    text=f"{current_page}/{total_pages}",
                    callback_data="page:noop"
                )
            )
            if current_page < total_pages:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="Наступні →",
                        callback_data=f"page:next:{context_key}"
                    )
                )
            rows.append(pagination_buttons)

        # Context buttons for quotes
        if quotes:
            context_buttons = []
            for i, quote in enumerate(quotes[:3], 1):
                meta = quote.get("metadata", {})
                msg_id = meta.get("message_id")
                if msg_id:
                    context_buttons.append(
                        InlineKeyboardButton(
                            text=f"📜 Контекст [{i}]",
                            callback_data=f"ctx:{msg_id}"
                        )
                    )
            if context_buttons:
                rows.append(context_buttons)

        return InlineKeyboardMarkup(rows) if rows else None

    def _build_full_keyboard(
        self,
        quotes: list[dict],
        context_key: str,
        current_page: int = 1,
        total_pages: int = 1,
        show_pagination: bool = False,
        suggestions: list[str] = None,
        state=None
    ) -> InlineKeyboardMarkup:
        """Build full keyboard with context, pagination, and suggestions."""
        rows = []

        # Suggestion buttons (if available)
        if suggestions and state:
            suggestion_buttons = []
            for suggestion in suggestions[:2]:  # Max 2 suggestions per row
                # Truncate for button display
                display = suggestion[:25] + "..." if len(suggestion) > 25 else suggestion
                # Get or create hash for this suggestion
                hash_str = None
                for h, q in state.suggestion_hashes.items():
                    if q == suggestion:
                        hash_str = h
                        break
                if not hash_str:
                    hash_str = state.add_suggestion(suggestion)
                suggestion_buttons.append(
                    InlineKeyboardButton(
                        text=f"💡 {display}",
                        callback_data=f"suggest:{hash_str}:{context_key}"
                    )
                )
            if suggestion_buttons:
                rows.append(suggestion_buttons)

        # Pagination row (if multiple pages)
        if show_pagination and total_pages > 1:
            pagination_buttons = []
            if current_page > 1:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="← Попередні",
                        callback_data=f"page:prev:{context_key}"
                    )
                )
            pagination_buttons.append(
                InlineKeyboardButton(
                    text=f"{current_page}/{total_pages}",
                    callback_data="page:noop"
                )
            )
            if current_page < total_pages:
                pagination_buttons.append(
                    InlineKeyboardButton(
                        text="Наступні →",
                        callback_data=f"page:next:{context_key}"
                    )
                )
            rows.append(pagination_buttons)

        # Context buttons for quotes
        if quotes:
            context_buttons = []
            for i, quote in enumerate(quotes[:3], 1):
                meta = quote.get("metadata", {})
                msg_id = meta.get("message_id")
                if msg_id:
                    context_buttons.append(
                        InlineKeyboardButton(
                            text=f"📜 Контекст [{i}]",
                            callback_data=f"ctx:{msg_id}"
                        )
                    )
            if context_buttons:
                rows.append(context_buttons)

        if not rows:
            return None

        return InlineKeyboardMarkup(rows)

    async def handle_suggestion_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle suggestion button clicks."""
        query = update.callback_query

        data = query.data
        if not data.startswith("suggest:"):
            return

        try:
            parts = data.split(":")
            hash_str = parts[1]
            context_key = parts[2]

            state = self.conversation_context.get_by_context_key(context_key)
            if not state:
                await query.answer("⚠️ Контекст сесії застарів", show_alert=True)
                return

            # Get suggestion query by hash
            suggestion_query = state.get_suggestion_by_hash(hash_str)
            if not suggestion_query:
                await query.answer("⚠️ Підказку не знайдено", show_alert=True)
                return

            await query.answer("🔍 Шукаю...")

            # Send new search status
            status_msg = await query.message.reply_text(f"🔍 Шукаю: \"{suggestion_query}\"...")

            # Get aliases and users
            aliases_dict = self.db.get_aliases_dict()
            users = self.db.get_all_users()

            # Parse the suggestion as a new question
            parsed = await self.question_parser.parse_async(suggestion_query, aliases_dict, users)

            # Perform search with agent
            if parsed.mentioned_users:
                all_results = []
                for user_id, username in parsed.mentioned_users:
                    results, session = await self.search_agent.search(
                        question=parsed.original_question,
                        initial_query=parsed.search_query,
                        user_id=user_id,
                        min_relevant=3,
                        max_iterations=3
                    )
                    all_results.extend(results)
            else:
                all_results, session = await self.search_agent.search(
                    question=parsed.original_question,
                    initial_query=parsed.search_query,
                    min_relevant=3,
                    max_iterations=3
                )

            # Synthesize answer
            synthesized = await self.answer_synthesizer.synthesize_async(
                question=parsed.original_question,
                results=all_results,
                mentioned_users=parsed.mentioned_users,
                filter_relevance=False
            )

            # Format response
            response = self.formatter.format_synthesized_answer(
                question=parsed.original_question,
                synthesized=synthesized,
                mentioned_users=parsed.mentioned_users
            )

            # Build keyboard
            new_context_key = self.conversation_context.make_context_key(
                query.message.chat_id, status_msg.message_id
            )
            new_state = self.conversation_context.store(
                chat_id=query.message.chat_id,
                bot_message_id=status_msg.message_id,
                original_question=parsed.original_question,
                search_query=parsed.search_query,
                mentioned_users=parsed.mentioned_users,
                results=synthesized.supporting_quotes[:5],
                all_results=all_results
            )

            keyboard = self._build_context_keyboard(
                quotes=synthesized.supporting_quotes,
                context_key=new_context_key,
                current_page=1,
                total_pages=new_state.total_pages,
                show_pagination=len(all_results) > 5
            )

            # Edit status message with response
            try:
                if len(response) <= 4096:
                    await status_msg.edit_text(response, reply_markup=keyboard)
                else:
                    await status_msg.edit_text(response[:4000] + "\n\n⚠️ Відповідь скорочено...", reply_markup=keyboard)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Suggestion callback error: {e}")
            await query.answer(f"❌ Помилка: {e}", show_alert=True)

    async def handle_thread_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle thread button clicks to show conversation thread."""
        query = update.callback_query

        data = query.data
        if not data.startswith("thread:"):
            return

        try:
            message_id = int(data.split(":")[1])
            await query.answer("🧵 Завантажую розмову...")

            # Get the full thread
            thread_data = self.db.get_full_thread(message_id)

            if not thread_data.get("messages"):
                await query.message.reply_text(
                    f"❌ Не вдалося знайти розмову для повідомлення #{message_id}"
                )
                return

            # Generate thread summary for longer threads
            summary = None
            messages = thread_data.get("messages", [])
            if len(messages) >= 5:
                try:
                    # Format messages for summary
                    summary_prompt = self.formatter.format_thread_summary_prompt(messages)

                    # Generate summary using chat service
                    from search.embeddings import ChatService
                    chat_service = ChatService()
                    summary = await chat_service.complete_async(
                        prompt=f"Summarize this {len(messages)}-message conversation in 2-3 bullet points. Use Ukrainian:\n\n{summary_prompt}",
                        system="You are a conversation summarizer. Provide concise, informative summaries in Ukrainian. Use bullet points.",
                        max_tokens=200
                    )
                except Exception as e:
                    logger.debug(f"Thread summary generation failed: {e}")

            # Format thread response
            response = self.formatter.format_thread(
                thread_data=thread_data,
                summary=summary
            )

            # Send as new message (threads can be long)
            if len(response) <= 4096:
                await query.message.reply_text(response)
            else:
                # Split into multiple messages if too long
                await query.message.reply_text(response[:4000] + "\n\n⚠️ Показано частину розмови...")

        except Exception as e:
            logger.error(f"Thread callback error: {e}")
            await query.answer(f"❌ Помилка: {e}", show_alert=True)

    async def _get_related_context(self, target, message_id: int, max_context: int = 10) -> list:
        """Get context messages filtered by topic relevance."""
        if not target.text or len(target.text.strip()) < 10:
            # For very short messages, fall back to chronological context
            return self.db.get_thread_context(message_id, before=5, after=5)

        # Get a wider range of messages chronologically (±100 message IDs)
        wide_context = self.db.get_thread_context(message_id, before=100, after=100)
        if len(wide_context) <= max_context:
            return wide_context

        # Use vector search to find messages related to target's topic
        # Search within the time range of the wide context
        try:
            related_results = self.vector_store.search(
                query=target.text,
                n_results=max_context * 2,  # Get more, then filter by ID range
                min_similarity=0.3
            )

            # Get message IDs that are both related AND in our time window
            wide_context_ids = {m.message_id for m in wide_context}
            related_ids = set()
            for r in related_results:
                msg_id = r.get("metadata", {}).get("message_id")
                if msg_id and msg_id in wide_context_ids:
                    related_ids.add(msg_id)

            # Always include the target message
            related_ids.add(message_id)

            # Filter wide_context to only related messages
            if related_ids:
                context_msgs = [m for m in wide_context if m.message_id in related_ids]
                # Sort by message_id to maintain chronological order
                context_msgs.sort(key=lambda m: m.message_id)
                return context_msgs[:max_context]

        except Exception as e:
            logger.warning(f"Vector search for context failed: {e}")

        # Fallback to chronological context
        return self.db.get_thread_context(message_id, before=5, after=5)

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

        # Send status message that we'll edit later
        status_msg = await message.reply_text("🔍 Аналізую уточнення...")

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

            # Edit status message with response (or send new if too long/edit fails)
            try:
                if len(response) <= 4096:
                    await status_msg.edit_text(response)
                    sent_message = status_msg
                else:
                    truncated = response[:4000] + "\n\n⚠️ Відповідь скорочено..."
                    await status_msg.edit_text(truncated)
                    sent_message = status_msg
            except Exception as edit_error:
                logger.warning(f"Failed to edit status message: {edit_error}")
                await status_msg.delete()
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
            # Try to edit status message with error, or send new
            try:
                await status_msg.edit_text(f"❌ Помилка: {e}")
            except Exception:
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

    # === Entity Alias Commands (зе → Зеленський, порох → Порошенко) ===

    async def entity_aliases(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /entity_aliases command - list all entity aliases."""
        try:
            # Get database aliases
            db_aliases = self.db.get_all_entity_aliases()

            # Format response
            lines = ["📚 **Аліаси сутностей** (сленг → канонічна форма)\n"]

            # Group by category
            by_category: dict[str, list] = {}
            for alias in db_aliases:
                cat = alias.category or "інше"
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(alias)

            # Count hardcoded
            hardcoded_count = len(HARDCODED_ALIASES)
            db_count = len(db_aliases)

            lines.append(f"📊 Всього: {len(ENTITY_ALIASES)} ({hardcoded_count} вбудованих + {db_count} користувацьких)\n")

            if db_aliases:
                lines.append("**Користувацькі аліаси:**")
                for cat, aliases in sorted(by_category.items()):
                    lines.append(f"\n_{cat.capitalize()}:_")
                    for a in aliases:
                        lines.append(f"  • {a.alias} → {a.canonical}")
            else:
                lines.append("_Користувацьких аліасів ще немає._")

            lines.append("\n**Команди:**")
            lines.append("/entity_alias <аліас> <канонічна форма> [категорія]")
            lines.append("/entity_alias_remove <аліас>")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Entity aliases error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def add_entity_alias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /entity_alias command - add entity alias."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Вкажіть аліас та канонічну форму\n\n"
                "Приклад: /entity_alias зелупа Зеленський політик\n"
                "Приклад: /entity_alias шиба shiba крипта"
            )
            return

        alias = context.args[0].lower()
        canonical = context.args[1]
        category = context.args[2] if len(context.args) > 2 else None
        user_id = update.message.from_user.id

        try:
            # Check if it's a hardcoded alias
            if alias in HARDCODED_ALIASES:
                await update.message.reply_text(
                    f"❌ Аліас '{alias}' вже вбудований (→ {HARDCODED_ALIASES[alias]})"
                )
                return

            success, message = self.db.add_entity_alias(
                alias=alias,
                canonical=canonical,
                category=category,
                added_by=user_id
            )

            if success:
                # Reload aliases to apply changes
                reload_entity_aliases()
                await update.message.reply_text(f"✅ {message}")
            else:
                await update.message.reply_text(f"❌ {message}")

        except Exception as e:
            logger.error(f"Add entity alias error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")

    async def remove_entity_alias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /entity_alias_remove command."""
        if not context.args:
            await update.message.reply_text(
                "❌ Вкажіть аліас для видалення\n"
                "Приклад: /entity_alias_remove зелупа"
            )
            return

        alias = context.args[0].lower()
        user_id = update.message.from_user.id

        try:
            # Check if it's a hardcoded alias
            if alias in HARDCODED_ALIASES:
                await update.message.reply_text(
                    f"❌ Аліас '{alias}' вбудований і не може бути видалений"
                )
                return

            success, message = self.db.remove_entity_alias(alias, user_id)

            if success:
                # Reload aliases to apply changes
                reload_entity_aliases()
                await update.message.reply_text(f"✅ {message}")
            else:
                await update.message.reply_text(f"❌ {message}")

        except Exception as e:
            logger.error(f"Remove entity alias error: {e}")
            await update.message.reply_text(f"❌ Помилка: {e}")


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
    app.add_handler(CommandHandler("mystats", handlers.mystats))
    app.add_handler(CommandHandler("upload", handlers.upload))

    # User alias management (nicknames for chat users)
    app.add_handler(CommandHandler("aliases", handlers.aliases))
    app.add_handler(CommandHandler("alias", handlers.add_alias))
    app.add_handler(CommandHandler("alias_remove", handlers.remove_alias))
    app.add_handler(CommandHandler("seed_aliases", handlers.seed_aliases))

    # Entity alias management (зе → Зеленський, порох → Порошенко)
    app.add_handler(CommandHandler("entity_aliases", handlers.entity_aliases))
    app.add_handler(CommandHandler("entity_alias", handlers.add_entity_alias))
    app.add_handler(CommandHandler("entity_alias_remove", handlers.remove_entity_alias))

    # Mention handler (for @bot questions)
    mention_filter = filters.TEXT & filters.Regex(f"(?i)@{BOT_USERNAME}")
    app.add_handler(MessageHandler(mention_filter, handlers.handle_mention))

    # Follow-up handler (replies to bot messages)
    reply_filter = filters.TEXT & filters.REPLY
    app.add_handler(MessageHandler(reply_filter, handlers.handle_followup))

    # Document upload handler (for upload wizard)
    app.add_handler(MessageHandler(filters.Document.ZIP, handlers.handle_document_upload))

    # Callback handlers for inline buttons
    app.add_handler(CallbackQueryHandler(handlers.handle_context_callback, pattern=r"^ctx:"))
    app.add_handler(CallbackQueryHandler(handlers.handle_pagination_callback, pattern=r"^page:"))
    app.add_handler(CallbackQueryHandler(handlers.handle_time_filter_callback, pattern=r"^time:"))
    app.add_handler(CallbackQueryHandler(handlers.handle_suggestion_callback, pattern=r"^suggest:"))
    app.add_handler(CallbackQueryHandler(handlers.handle_thread_callback, pattern=r"^thread:"))
    app.add_handler(CallbackQueryHandler(handlers.handle_upload_callback, pattern=r"^upload:"))

    return handlers
