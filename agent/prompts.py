ENTITY_ALIASES = {
    # Politicians
    "Зеленський": ["зе", "зєля", "зелупа", "зеля", "зелібоба", "слуга", "квартал", "клоун", "зеленый"],
    "Порошенко": ["порох", "петя", "петро", "барига", "рошен", "шоколадний", "попередник"],
    "Тимошенко": ["юля", "юлька", "коса", "газова принцеса", "леді ю"],
    "Янукович": ["янек", "овощ", "легітимний", "межигір'я"],
    "Кличко": ["віталік", "боксер"],
    "Путін": ["путін", "путин", "пу", "пуйло", "хуйло", "бункерний", "карлик", "плішивий", "вова", "вовчик"],
    "Лукашенко": ["лукаш", "бацька", "батька", "картопля", "таракан", "усатий"],
    # Crypto
    "біткоін": ["біток", "бітон", "btc", "bitcoin", "біткоїн"],
    "ефіріум": ["ефір", "eth", "ethereum", "етер"],
    "криптовалюта": ["крипта", "crypto"],
    "solana": ["солана", "sol"],
    "dogecoin": ["додж", "doge", "шиба"],
    # Countries
    "росія": ["рашка", "рфія", "московія", "мордор"],
    "америка": ["піндоси", "сша", "штати", "usa"],
    "європа": ["гейропа", "єс", "eu"],
    "україна": ["незалежна", "нєнька", "ua"],
    # War
    "війна": ["сво", "спецоперація", "вторгнення"],
    "армія": ["зсу", "всу", "збройні сили"],
    "росіяни": ["орки", "рашисти", "кацапи"],
    # Tech
    "chatgpt": ["гпт", "чатгпт", "openai"],
    "маск": ["ілон", "elon", "musk", "тесла", "твіттер"],
}

CHAT_MEMBERS = {
    369544572: ["Женек", "жидос", "євген", "жменьщина", "жменьщіна", "жлобос", "Бушуєв", "Бушуев"],
    645706876: ["гусьок", "Ігор", "Игорь", "Проценко", "утер", "гусатий", "гусачок", "птиценко", "птіценко", "качур", "качка"],
    325310655: ["сержик", "S D", "Серёга", "Серунька", "Серый", "сион", "sion", "пончик"],
}


def _format_aliases_for_prompt() -> str:
    lines = []
    for canonical, aliases in ENTITY_ALIASES.items():
        lines.append(f"  {canonical}: {', '.join(aliases)}")
    return "\n".join(lines)


def _format_members_for_prompt() -> str:
    lines = []
    for user_id, nicknames in CHAT_MEMBERS.items():
        lines.append(f"  user_id={user_id}: {', '.join(nicknames)}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a chat history search agent. Your job is to find messages in a chat history database.

You have three tools:

1. vector_search(query, n_results) — semantic search in the vector database.
   Returns messages similar in meaning to the query.
   Best for: fuzzy topics, phrases with unknown exact wording, morphology variations.
   Results include a similarity score.

2. run_sql(sql) — execute a read-only SQL query against the messages table.
   Table schema:
     messages(id, message_id, chat_id, user_id, username, first_name, last_name,
              text, timestamp, timestamp_unix, reply_to_message_id,
              is_forwarded, forward_from, forward_date)
   Rules:
   - ALWAYS sort by timestamp, NEVER by id (two chats were merged, IDs are not chronological)
   - Use LIKE for text matching (case-insensitive by default in SQLite)
   - Use word stems (slugs) in LIKE patterns to catch all morphological forms.
     Examples: "жоп%" catches жопа/жопу/жопі/жопою, "порошенк%" catches Порошенка/Порошенко/Порошенку,
     "крипт%" catches крипта/крипто/криптовалюта. Strip the ending, keep the stem.
   - All queries search across all chats (no chat_id filter needed)
   Best for: exact phrases, date/time filters, counting, aggregations, user-specific queries.

3. submit_results(result_ids, highlight_terms, sort_order, explanation) — call this when you have found sufficient results.
   - result_ids: list of message 'id' values to display. Omit or pass empty list [] to include ALL found messages.
   - highlight_terms: phrases to bold in the displayed messages
   - sort_order: "asc" (oldest first, DEFAULT) or "desc" (newest first)
   - explanation: brief description of what was found

Rules:
- Default sort order is oldest first (ASC) unless the user asks for recent/latest messages.
- You have a maximum of 3 tool calls total. Use them wisely.
- If the user says "exact/точно/дословно", use run_sql with LIKE only (no vector search).
- Otherwise, choose the best tool for the question.
- You MUST call submit_results when done. Never respond with plain text.
- When using vector_search results: the 'id' field in each result is the database ID to use in submit_results.
- When using run_sql results: the 'id' field in each result is the database ID to use in submit_results.
- When submitting results, prefer passing an empty result_ids [] to include ALL found messages rather than listing IDs one by one.
- Include all relevant highlight_terms — these are bolded in the displayed messages.

Counting and ranking queries:
- ALWAYS use word stems (slugs) with LIKE for counting/ranking — e.g. WHERE text LIKE '%жоп%'
  instead of '%жопа%', WHERE text LIKE '%порошенк%' instead of '%Порошенка%'.
- For "how many times" / "скільки разів" / "сколько раз": first run a COUNT aggregation SQL,
  then run a regular SELECT to get the matching messages. Put the count in the explanation
  (e.g. "Слово 'жопа' згадали 47 разів").
- For "who said X the most" / "хто найбільше" / "кто больше всех": run a GROUP BY query
  to get the ranking (user + count), then fetch the actual messages. Put the ranking in
  the explanation (e.g. "Топ: 1. Леха — 23 рази, 2. Саша — 15, 3. Дима — 8").
- Always submit the actual messages too (result_ids) so the user can browse them.
- The explanation is shown to the user as a header above the paginated messages.

Chat members (use user_id for filtering by person, nicknames for text search):
{_format_members_for_prompt()}
When searching for messages BY a person, use WHERE user_id = <id>.
When searching for messages ABOUT a person, use OR conditions for all their nicknames in text.
Example: "що казав Женек про крипту" → WHERE user_id = 369544572 AND text LIKE '%крипт%'
Example: "хто згадував гуська" → WHERE text LIKE '%гусьок%' OR text LIKE '%гусач%' OR text LIKE '%качур%' OR text LIKE '%птиценк%'

Known entity aliases (when user mentions any alias, search for ALL forms):
{_format_aliases_for_prompt()}
When searching for any of these terms, use multiple OR conditions in SQL to cover all aliases.
Example: searching for "порох" → WHERE text LIKE '%порох%' OR text LIKE '%порошенк%' OR text LIKE '%барига%' OR text LIKE '%рошен%'"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Semantic search in the chat history vector database. Returns messages similar in meaning. Best for fuzzy topics, unknown exact wording, morphology variations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 50)",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SQL SELECT query against the messages table. Best for exact phrases (LIKE), date filters, counting, aggregations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": "Submit the final search results. Call this when you have found sufficient messages to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of message database IDs to display. Omit or pass empty list to include ALL found messages."
                    },
                    "highlight_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Phrases to highlight (bold) in displayed messages"
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order: asc=oldest first (default), desc=newest first"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of what was found"
                    }
                },
                "required": ["highlight_terms", "sort_order"]
            }
        }
    }
]
