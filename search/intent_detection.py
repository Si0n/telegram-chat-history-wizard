"""
Intent detection for search queries.
Classifies questions into different intents for optimized search strategies.
"""
import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class SearchIntent(Enum):
    """Types of search intents."""
    QUOTE_SEARCH = "quote_search"    # Find exact quotes
    YES_NO = "yes_no"                # Did X say Y?
    SUMMARY = "summary"              # What do people think about X?
    COMPARISON = "comparison"        # Compare opinions
    ANALYTICS = "analytics"          # Who talks most? Statistics
    GENERAL = "general"              # Default fallback


@dataclass
class SearchStrategy:
    """Strategy parameters for different intents."""
    intent: SearchIntent
    min_results: int = 3
    max_results: int = 10
    use_hyde: bool = True
    use_reranking: bool = True
    relevance_threshold: int = 5
    require_exact_match: bool = False
    aggregate_by_user: bool = False


# Intent detection patterns (multilingual: Ukrainian, Russian, English)
INTENT_PATTERNS = {
    SearchIntent.YES_NO: [
        # Ukrainian
        r'\bчи\s+\w+\s+каза',
        r'\bчи\s+\w+\s+говори',
        r'\bчи\s+\w+\s+писа',
        r'\bчи\s+\w+\s+вважа',
        r'\bчи\s+\w+\s+дума',
        r'\bчи\s+було\b',
        r'\bчи\s+є\b',
        # Russian
        r'\bговорил\s+ли\b',
        r'\bсказал\s+ли\b',
        r'\bписал\s+ли\b',
        r'\bбыло\s+ли\b',
        # English
        r'\bdid\s+\w+\s+say\b',
        r'\bdid\s+\w+\s+mention\b',
        r'\bhas\s+\w+\s+ever\b',
        r'\bwas\s+there\b',
    ],
    SearchIntent.ANALYTICS: [
        # Quantitative patterns
        r'хто\s+більше',
        r'хто\s+частіше',
        r'скільки\s+раз',
        r'найактивніш',
        r'хто\s+найбільше',
        r'кто\s+больше',
        r'кто\s+чаще',
        r'сколько\s+раз',
        r'самый\s+активный',
        r'who\s+talks?\s+more',
        r'who\s+speaks?\s+more',
        r'how\s+many\s+times',
        r'most\s+active',
        r'топ[\s-]?\d+',
        r'top[\s-]?\d+',
        # Mention/count patterns
        r'хто\s+згадував',
        r'хто\s+писав\s+про',
        r'кто\s+упоминал',
        r'who\s+mentioned',
        # User-specific message queries
        r'покажи\s+повідомлення\s+від',
        r'повідомлення\s+(від\s+)?user',
        r'повідомлення\s+користувача',
        r'що\s+писав\s+user',
        r'що\s+писала?\s+user',
        r'сообщения\s+(от\s+)?user',
        r'сообщения\s+пользователя',
        r'что\s+писал\s+user',
        r'messages\s+from\s+user',
        r'show\s+messages\s+from',
        r'user#?\d+',  # Match User#123 or User123 patterns
        # Behavioral/Qualitative patterns
        r'хто\s+най\w+ший',
        r'кто\s+самый',
        r'who\s+is\s+(the\s+)?(more|most)\s+\w+',
        r'хто\s+більш\s+\w+',
        r'кто\s+более',
        r'хто\s+злий',
        r'хто\s+добрий',
        r'хто\s+строгий',
        r'хто\s+лається',
        r'хто\s+матюкається',
        r'хто\s+позитивн',
        r'хто\s+негативн',
        r'кто\s+злой',
        r'кто\s+добрый',
        r'кто\s+ругается',
    ],
    SearchIntent.COMPARISON: [
        # Ukrainian
        r'порівня',
        r'різниця\s+між',
        r'хто\s+з\s+них',
        r'\bvs\b',
        # Russian
        r'сравни',
        r'разница\s+между',
        r'кто\s+из\s+них',
        # English
        r'compare',
        r'difference\s+between',
        r'versus',
    ],
    SearchIntent.SUMMARY: [
        # Ukrainian
        r'що\s+думають',
        r'яка\s+думка',
        r'загалом\s+про',
        r'підсум',
        # Russian
        r'что\s+думают',
        r'какое\s+мнение',
        r'в\s+целом\s+про',
        r'резюм',
        # English
        r'what\s+do\s+(people|they)\s+think',
        r'general\s+opinion',
        r'summarize',
        r'summary\s+of',
    ],
    SearchIntent.QUOTE_SEARCH: [
        # Ukrainian
        r'цитат[аи]',
        r'дослівно',
        r'точн[іе]\s+слова',
        # Russian
        r'цитат[ау]',
        r'дословно',
        r'точные\s+слова',
        # English
        r'exact\s+quote',
        r'verbatim',
        r'word\s+for\s+word',
    ],
}

# Default strategies for each intent
INTENT_STRATEGIES = {
    SearchIntent.QUOTE_SEARCH: SearchStrategy(
        intent=SearchIntent.QUOTE_SEARCH,
        min_results=1,
        max_results=5,
        use_hyde=False,  # Want exact matches, not hypotheticals
        use_reranking=True,
        relevance_threshold=7,
        require_exact_match=True
    ),
    SearchIntent.YES_NO: SearchStrategy(
        intent=SearchIntent.YES_NO,
        min_results=1,
        max_results=5,
        use_hyde=True,
        use_reranking=True,
        relevance_threshold=6
    ),
    SearchIntent.SUMMARY: SearchStrategy(
        intent=SearchIntent.SUMMARY,
        min_results=5,
        max_results=15,
        use_hyde=True,
        use_reranking=True,
        relevance_threshold=5,
        aggregate_by_user=True
    ),
    SearchIntent.COMPARISON: SearchStrategy(
        intent=SearchIntent.COMPARISON,
        min_results=3,
        max_results=10,
        use_hyde=True,
        use_reranking=True,
        relevance_threshold=5
    ),
    SearchIntent.ANALYTICS: SearchStrategy(
        intent=SearchIntent.ANALYTICS,
        min_results=0,  # Analytics may not need search results
        max_results=0,
        use_hyde=False,
        use_reranking=False,
        aggregate_by_user=True
    ),
    SearchIntent.GENERAL: SearchStrategy(
        intent=SearchIntent.GENERAL,
        min_results=3,
        max_results=10,
        use_hyde=True,
        use_reranking=True,
        relevance_threshold=5
    ),
}


def detect_intent(question: str) -> SearchIntent:
    """
    Detect the intent of a search question.

    Args:
        question: The user's question

    Returns:
        The detected SearchIntent
    """
    question_lower = question.lower()

    # Check patterns in order of specificity
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                logger.debug(f"Detected intent {intent.value} for: {question[:50]}...")
                return intent

    logger.debug(f"Using default GENERAL intent for: {question[:50]}...")
    return SearchIntent.GENERAL


def get_search_strategy(intent: SearchIntent) -> SearchStrategy:
    """
    Get the search strategy for a given intent.

    Args:
        intent: The SearchIntent

    Returns:
        SearchStrategy with parameters for this intent
    """
    return INTENT_STRATEGIES.get(intent, INTENT_STRATEGIES[SearchIntent.GENERAL])


def extract_analytics_type(question: str) -> Optional[str]:
    """
    Extract the type of analytics query from the question.

    Returns:
        "quantitative" for count-based, "behavioral" for trait-based, None if not analytics
    """
    question_lower = question.lower()

    # Behavioral markers (trait analysis)
    behavioral_markers = [
        'злий', 'злой', 'angry', 'mad',
        'строгий', 'strict',
        'позитивн', 'негативн', 'positive', 'negative',
        'психо', 'божевільн', 'crazy', 'сумасшедш',
        'лається', 'матюкається', 'swears', 'ругается',
        'добрий', 'добрый', 'kind', 'nice',
        'агресивн', 'агрессивн', 'aggressive',
        'токсичн', 'toxic',
    ]

    # Quantitative markers (counts)
    quantitative_markers = [
        'скільки', 'сколько', 'how many', 'count',
        'найактивніш', 'самый активный', 'most active',
        'частіше', 'чаще', 'more often',
        'більше пише', 'больше пишет', 'writes more',
        'згадував', 'упоминал', 'mentioned',
        'топ', 'top',
    ]

    if any(m in question_lower for m in behavioral_markers):
        return "behavioral"

    if any(m in question_lower for m in quantitative_markers):
        return "quantitative"

    return None


def extract_trait_from_question(question: str) -> Optional[str]:
    """
    Extract the behavioral trait being asked about.

    Args:
        question: The user's question

    Returns:
        The trait name or None
    """
    question_lower = question.lower()

    # Map keywords to trait names
    trait_keywords = {
        'angry': ['злий', 'злой', 'angry', 'mad', 'бісить', 'бесит', 'злиться'],
        'strict': ['строгий', 'strict', 'суровий', 'суровый'],
        'positive': ['позитивн', 'positive', 'оптиміст', 'оптимист', 'веселий', 'веселый'],
        'negative': ['негативн', 'negative', 'песиміст', 'пессимист'],
        'psycho': ['психо', 'божевільн', 'сумасшедш', 'crazy', 'шизо'],
        'swears': ['лається', 'матюкається', 'ругается', 'матерится', 'swears'],
        'kind': ['добрий', 'добрый', 'kind', 'nice', 'милий', 'милый'],
        'aggressive': ['агресивн', 'агрессивн', 'aggressive'],
        'toxic': ['токсичн', 'toxic'],
    }

    for trait, keywords in trait_keywords.items():
        if any(kw in question_lower for kw in keywords):
            return trait

    return None
