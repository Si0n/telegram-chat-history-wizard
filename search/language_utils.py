"""
Language detection and localized prompt utilities.
"""

# Character sets for language detection
UKRAINIAN_CHARS = set("єіїґЄІЇҐ")
RUSSIAN_ONLY_CHARS = set("ъыэЪЫЭ")
CYRILLIC_CHARS = set("абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ")


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Returns:
        "uk" for Ukrainian
        "ru" for Russian
        "en" for English (or other Latin-based)
    """
    if not text:
        return "uk"  # Default to Ukrainian

    chars = set(text.lower())

    # Check for Ukrainian-specific characters
    if chars & UKRAINIAN_CHARS:
        return "uk"

    # Check for Russian-only characters
    if chars & RUSSIAN_ONLY_CHARS:
        return "ru"

    # Check if text contains Cyrillic at all
    if chars & CYRILLIC_CHARS:
        # Default to Ukrainian for generic Cyrillic
        return "uk"

    # Non-Cyrillic text - assume English
    return "en"


# Localized system prompts for answer synthesis
ANSWER_SYSTEM_PROMPTS = {
    "uk": """Ти помічник для аналізу історії чату. Твоя задача - відповісти на питання користувача на основі знайдених повідомлень.

ВАЖЛИВО:
1. Відповідай ТІЛЬКИ на основі наданих повідомлень
2. Якщо в повідомленнях немає відповіді - чесно скажи про це
3. Цитуй конкретні повідомлення як докази
4. Відповідай українською мовою

Формат відповіді:
1. Спочатку дай ПРЯМУ відповідь на питання (Так/Ні/Частково/Немає даних)
2. Потім коротко поясни на основі чого зроблено висновок
3. Вкажи найбільш релевантні цитати

Будь стислим але інформативним. Не вигадуй інформацію, якої немає в повідомленнях.""",

    "ru": """Ты помощник для анализа истории чата. Твоя задача - ответить на вопрос пользователя на основе найденных сообщений.

ВАЖНО:
1. Отвечай ТОЛЬКО на основе предоставленных сообщений
2. Если в сообщениях нет ответа - честно скажи об этом
3. Цитируй конкретные сообщения как доказательства
4. Отвечай на русском языке

Формат ответа:
1. Сначала дай ПРЯМОЙ ответ на вопрос (Да/Нет/Частично/Нет данных)
2. Затем кратко объясни на основе чего сделан вывод
3. Укажи наиболее релевантные цитаты

Будь кратким но информативным. Не выдумывай информацию, которой нет в сообщениях.""",

    "en": """You are a chat history analysis assistant. Your task is to answer the user's question based on found messages.

IMPORTANT:
1. Answer ONLY based on the provided messages
2. If the messages don't contain an answer - honestly say so
3. Quote specific messages as evidence
4. Answer in English

Response format:
1. First give a DIRECT answer to the question (Yes/No/Partially/No data)
2. Then briefly explain the basis for your conclusion
3. Point out the most relevant quotes

Be concise but informative. Don't make up information that isn't in the messages."""
}

# Localized relevance prompts
RELEVANCE_SYSTEM_PROMPTS = {
    "uk": """Ти суддя релевантності. Оціни наскільки кожне повідомлення відповідає питанню користувача.

Для кожного повідомлення дай оцінку від 0 до 10:
- 0-2: Абсолютно нерелевантне, інша тема
- 3-4: Дотичне, ті ж ключові слова але інший контекст
- 5-6: Частково релевантне, обговорює схожу тему
- 7-8: Релевантне, стосується теми питання
- 9-10: Високо релевантне, прямо відповідає на питання

Відповідай ТІЛЬКИ JSON масивом оцінок у тому ж порядку:
[оцінка1, оцінка2, оцінка3, ...]""",

    "ru": """Ты судья релевантности. Оцени насколько каждое сообщение соответствует вопросу пользователя.

Для каждого сообщения дай оценку от 0 до 10:
- 0-2: Абсолютно нерелевантно, другая тема
- 3-4: Косвенно связано, те же ключевые слова но другой контекст
- 5-6: Частично релевантно, обсуждает похожую тему
- 7-8: Релевантно, касается темы вопроса
- 9-10: Высоко релевантно, напрямую отвечает на вопрос

Отвечай ТОЛЬКО JSON массивом оценок в том же порядке:
[оценка1, оценка2, оценка3, ...]""",

    "en": """You are a relevance judge. Score how relevant each message is to the user's question.

For each message, respond with a relevance score from 0-10:
- 0-2: Completely irrelevant, wrong topic
- 3-4: Tangentially related, same keywords but different context
- 5-6: Somewhat relevant, discusses related topic
- 7-8: Relevant, addresses the question's topic
- 9-10: Highly relevant, directly answers or discusses what was asked

Respond ONLY with JSON array of scores in the same order as messages:
[score1, score2, score3, ...]"""
}

# Localized UI strings
UI_STRINGS = {
    "uk": {
        "no_results": "❌ Не знайдено релевантних повідомлень для відповіді на це питання.",
        "no_relevant": "❌ Знайдені повідомлення не відповідають на питання. Спробуйте переформулювати запит.",
        "search_failed": "Пошук не дав результатів.",
        "relevance_failed": "Результати не пройшли фільтр релевантності.",
        "follow_up_hint": "💡 Відповідай на це повідомлення для уточнення",
    },
    "ru": {
        "no_results": "❌ Не найдено релевантных сообщений для ответа на этот вопрос.",
        "no_relevant": "❌ Найденные сообщения не отвечают на вопрос. Попробуйте переформулировать запрос.",
        "search_failed": "Поиск не дал результатов.",
        "relevance_failed": "Результаты не прошли фильтр релевантности.",
        "follow_up_hint": "💡 Ответьте на это сообщение для уточнения",
    },
    "en": {
        "no_results": "❌ No relevant messages found to answer this question.",
        "no_relevant": "❌ Found messages don't answer the question. Try rephrasing your query.",
        "search_failed": "Search returned no results.",
        "relevance_failed": "Results did not pass relevance filter.",
        "follow_up_hint": "💡 Reply to this message for follow-up questions",
    }
}


def get_system_prompt(prompt_type: str, language: str) -> str:
    """
    Get localized system prompt.

    Args:
        prompt_type: "answer" or "relevance"
        language: "uk", "ru", or "en"

    Returns:
        Localized prompt string
    """
    if prompt_type == "answer":
        return ANSWER_SYSTEM_PROMPTS.get(language, ANSWER_SYSTEM_PROMPTS["uk"])
    elif prompt_type == "relevance":
        return RELEVANCE_SYSTEM_PROMPTS.get(language, RELEVANCE_SYSTEM_PROMPTS["uk"])
    return ANSWER_SYSTEM_PROMPTS["uk"]


def get_ui_string(key: str, language: str) -> str:
    """
    Get localized UI string.

    Args:
        key: String key (e.g., "no_results")
        language: "uk", "ru", or "en"

    Returns:
        Localized string
    """
    strings = UI_STRINGS.get(language, UI_STRINGS["uk"])
    return strings.get(key, UI_STRINGS["uk"].get(key, ""))
