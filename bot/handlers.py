import asyncio
import logging
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import config
from db.database import Database
from agent.loop import AgentLoop
from agent.state import StateManager, SearchState, DialogueState
from agent.formatter import Formatter
from agent.dialogue import DialogueWindow

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, db: Database, agent: AgentLoop):
        self.db = db
        self.agent = agent
        self.state = StateManager()
        self.formatter = Formatter()
        self.dialogue = DialogueWindow(db)

    # --- Commands ---

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hi! I search through chat history.\n\n"
            "In a group — mention me with your question: @bot_name who said X?\n"
            "In DM — just type your question.\n\n"
            "Type /help for examples."
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "<b>How to search:</b>\n\n"
            "Ask a question — I'll find the messages.\n\n"
            "<b>Examples:</b>\n"
            "\u2022 Who first said 'разбирается в интеллекте'?\n"
            "\u2022 What did Леха say about crypto in 2023?\n"
            "\u2022 How many messages did Саша send?\n"
            "\u2022 What were we talking about last week?\n\n"
            "<b>Tips:</b>\n"
            "\u2022 Say 'точно' or 'дословно' for exact phrase match\n"
            "\u2022 Click \U0001f4ac buttons to see surrounding dialogue\n"
            "\u2022 Use \u2b05\ufe0f/\u27a1\ufe0f to navigate pages or scroll dialogue",
            parse_mode="HTML",
        )

    # --- Message Handling ---

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle search queries (group mentions and DMs)."""
        message = update.message
        if not message or not message.text:
            return

        # Extract query text
        query = message.text

        # Strip bot mention in groups
        if message.chat.type != "private" and context.bot.username:
            mention = f"@{context.bot.username}"
            if mention.lower() not in query.lower():
                return  # Not mentioned in group, ignore
            query = re.sub(re.escape(mention), "", query, flags=re.IGNORECASE).strip()

        if not query:
            return

        # Send "searching" indicator
        reply = await message.reply_text("\U0001f50d Searching...")

        # Run agent in thread (blocking I/O)
        try:
            result = await asyncio.to_thread(self.agent.process_query, query)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await reply.edit_text("Search service is temporarily unavailable. Try again later.")
            return

        # Handle errors
        if result.get("error"):
            await reply.edit_text(result["error"])
            return

        # Handle empty results
        if not result["results"]:
            await reply.edit_text("Nothing found. Try rephrasing your question.")
            return

        # Store state
        search_state = SearchState(
            all_results=result["results"],
            original_query=query,
            sort_order=result.get("sort_order", "asc"),
            highlight_terms=result.get("highlight_terms", []),
        )
        self.state.set(message.chat_id, reply.message_id, search_state)

        # Format first page
        page_results = result["results"][:config.RESULTS_PER_PAGE]
        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(result["results"]),
            page=0,
            highlight_terms=search_state.highlight_terms,
            sort_order=search_state.sort_order,
        )

        await reply.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    # --- Callback Handlers ---

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Route inline button callbacks."""
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("p:"):
            await self._handle_page(query, data)
        elif data.startswith("d:"):
            await self._handle_dialogue_open(query, data)
        elif data.startswith("db:"):
            await self._handle_dialogue_back(query, data)
        elif data.startswith("df:"):
            await self._handle_dialogue_forward(query, data)
        elif data == "br":
            await self._handle_back_to_results(query)

    async def _handle_page(self, query, data: str):
        """Navigate search result pages."""
        page = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, SearchState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        state.current_page = page
        per_page = config.RESULTS_PER_PAGE
        start = page * per_page
        page_results = state.all_results[start : start + per_page]

        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(state.all_results),
            page=page,
            highlight_terms=state.highlight_terms,
            sort_order=state.sort_order,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_open(self, query, data: str):
        """Open dialogue window around a message."""
        msg_id = int(data.split(":")[1])
        chat_id = query.message.chat_id
        bot_msg_id = query.message.message_id

        # Save search state before overwriting with dialogue state
        search_state = self.state.get(chat_id, bot_msg_id)
        saved_search = search_state if isinstance(search_state, SearchState) else None
        highlight_terms = search_state.highlight_terms if search_state else []

        # Open dialogue
        window, anchor_id, anchor_chat_id = await asyncio.to_thread(
            self.dialogue.open, msg_id
        )

        if not window:
            await query.message.edit_text("Message not found.")
            return

        # Check navigation availability
        has_earlier = await asyncio.to_thread(
            self.dialogue.has_earlier, anchor_chat_id, window[0].get("timestamp_unix", 0)
        )
        has_later = await asyncio.to_thread(
            self.dialogue.has_later, anchor_chat_id, window[-1].get("timestamp_unix", 0)
        )

        # Store dialogue state with embedded search state for "back to results"
        dialogue_state = DialogueState(
            anchor_message_id=anchor_id,
            anchor_chat_id=anchor_chat_id,
            current_window=window,
            highlight_terms=highlight_terms,
            saved_search_state=saved_search,
        )
        self.state.set(chat_id, bot_msg_id, dialogue_state)

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=anchor_id,
            highlight_terms=highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_back(self, query, data: str):
        """Scroll dialogue backward."""
        first_ts = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, DialogueState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Fetch earlier messages
        earlier = await asyncio.to_thread(
            self.dialogue.scroll_back, state.anchor_chat_id, first_ts
        )

        if not earlier:
            return  # No earlier messages

        # New window: earlier + first message of current window (overlap)
        window = earlier + [state.current_window[0]]
        state.current_window = window

        has_earlier = await asyncio.to_thread(
            self.dialogue.has_earlier, state.anchor_chat_id, window[0].get("timestamp_unix", 0)
        )
        has_later = True  # We came from a later window, so there are later messages

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=state.anchor_message_id,
            highlight_terms=state.highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_forward(self, query, data: str):
        """Scroll dialogue forward."""
        last_ts = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, DialogueState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Fetch later messages
        later = await asyncio.to_thread(
            self.dialogue.scroll_forward, state.anchor_chat_id, last_ts
        )

        if not later:
            return  # No later messages

        # New window: last message of current window (overlap) + later
        window = [state.current_window[-1]] + later
        state.current_window = window

        has_earlier = True  # We came from an earlier window
        has_later = await asyncio.to_thread(
            self.dialogue.has_later, state.anchor_chat_id, window[-1].get("timestamp_unix", 0)
        )

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=state.anchor_message_id,
            highlight_terms=state.highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_back_to_results(self, query):
        """Return to search results from dialogue view."""
        chat_id = query.message.chat_id
        bot_msg_id = query.message.message_id

        state = self.state.get(chat_id, bot_msg_id)

        # Recover search state embedded in dialogue state
        search_state = None
        if isinstance(state, DialogueState) and state.saved_search_state:
            search_state = state.saved_search_state

        if not isinstance(search_state, SearchState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Restore search state to this message
        self.state.set(chat_id, bot_msg_id, search_state)

        page = search_state.current_page
        per_page = config.RESULTS_PER_PAGE
        start = page * per_page
        page_results = search_state.all_results[start : start + per_page]

        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(search_state.all_results),
            page=page,
            highlight_terms=search_state.highlight_terms,
            sort_order=search_state.sort_order,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)


def setup_handlers(app: Application, db: Database, agent: AgentLoop):
    """Register all handlers with the Telegram application."""
    handlers = BotHandlers(db, agent)

    app.add_handler(CommandHandler("start", handlers.cmd_start))
    app.add_handler(CommandHandler("help", handlers.cmd_help))
    app.add_handler(CallbackQueryHandler(handlers.handle_callback))

    # DM: all text messages are queries
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
        handlers.handle_message,
    ))

    # Group: messages mentioning the bot
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
        handlers.handle_message,
    ))
