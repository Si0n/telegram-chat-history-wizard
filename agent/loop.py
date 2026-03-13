import json
import logging

import chromadb
from chromadb.config import Settings
from openai import OpenAI

import config
from db.database import Database
from .prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class AgentLoop:
    """Orchestrates LLM <-> tool execution for chat history search."""

    def __init__(self, db: Database):
        self.db = db
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.collection = None
        try:
            chroma = chromadb.PersistentClient(
                path=str(config.CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = chroma.get_collection("messages")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable, SQL-only mode: {e}")

    def process_query(self, user_message: str) -> dict:
        """
        Run the agent loop synchronously.

        Returns:
            {
                "results": [list of message dicts with id, text, first_name, timestamp, chat_id, ...],
                "highlight_terms": [...],
                "sort_order": "asc" | "desc",
                "explanation": "...",
                "error": None | "error message"
            }
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        collected_results: dict[int, dict] = {}  # id -> message dict

        for iteration in range(config.AGENT_MAX_ITERATIONS):
            try:
                response = self.openai.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._error_response("Search service is temporarily unavailable. Try again later.")

            choice = response.choices[0]

            # No tool calls — LLM responded with text (fallback)
            if not choice.message.tool_calls:
                return self._fallback_response(collected_results, choice.message.content)

            # Process tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "submit_results":
                    return self._handle_submit(args, collected_results)
                elif name == "vector_search":
                    result = self._exec_vector_search(args)
                elif name == "run_sql":
                    result = self._exec_sql(args)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                # Accumulate results
                if isinstance(result, list):
                    for r in result:
                        if "id" in r and r["id"] not in collected_results:
                            collected_results[r["id"]] = r

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        # Max iterations exhausted
        return self._fallback_response(collected_results)

    # --- Tool Executors ---

    def _exec_vector_search(self, args: dict) -> list[dict] | dict:
        if not self.collection:
            return {"error": "Vector search unavailable. Use run_sql instead."}

        query = args.get("query", "")
        n_results = min(args.get("n_results", 20), config.AGENT_MAX_RESULTS)

        try:
            # Embed query
            emb_response = self.openai.embeddings.create(
                model=config.EMBEDDING_MODEL, input=query
            )
            query_embedding = emb_response.data[0].embedding

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return [{"error": f"Vector search failed: {e}"}]

        if not results["ids"] or not results["ids"][0]:
            return []

        # Map vector IDs to DB ids, fetch full messages
        matches = []
        for i, vec_id in enumerate(results["ids"][0]):
            try:
                db_id = int(vec_id.split("_")[1])
            except (IndexError, ValueError):
                continue

            similarity = round(1 - results["distances"][0][i], 3)
            msg = self.db.get_message_by_db_id(db_id)  # Returns dict
            if not msg:
                continue

            msg["similarity"] = similarity
            matches.append(msg)

        return matches

    def _exec_sql(self, args: dict) -> list[dict] | dict:
        sql = args.get("sql", "")
        try:
            return self.db.execute_safe_sql(sql)
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {"error": f"SQL error: {e}"}

    # --- Response Builders ---

    def _handle_submit(self, args: dict, collected: dict) -> dict:
        result_ids = args.get("result_ids", [])
        highlight_terms = args.get("highlight_terms", [])
        sort_order = args.get("sort_order", "asc")
        explanation = args.get("explanation", "")

        # Fetch messages that were in collected results
        results = [collected[rid] for rid in result_ids if rid in collected]

        # If some IDs weren't in collected (e.g., from SQL), fetch from DB
        missing_ids = [rid for rid in result_ids if rid not in collected]
        if missing_ids:
            db_msgs = self.db.get_messages_by_db_ids(missing_ids)
            results.extend(db_msgs)  # Already plain dicts from _msg_to_dict

        # Sort by timestamp
        reverse = sort_order == "desc"
        results.sort(key=lambda r: r.get("timestamp") or "", reverse=reverse)

        return {
            "results": results,
            "highlight_terms": highlight_terms,
            "sort_order": sort_order,
            "explanation": explanation,
            "error": None,
        }

    def _fallback_response(self, collected: dict, explanation: str = "") -> dict:
        results = list(collected.values())
        results.sort(key=lambda r: r.get("timestamp") or "")
        return {
            "results": results,
            "highlight_terms": [],
            "sort_order": "asc",
            "explanation": explanation or "Search completed",
            "error": None,
        }

    def _error_response(self, message: str) -> dict:
        return {
            "results": [],
            "highlight_terms": [],
            "sort_order": "asc",
            "explanation": "",
            "error": message,
        }
