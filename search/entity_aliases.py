"""
Entity aliases for public figures, places, and common terms.
Helps search understand slang, nicknames, and abbreviations.

Aliases are loaded from:
1. Hardcoded defaults (HARDCODED_ALIASES below)
2. User-added aliases from database (loaded at runtime)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Format: "alias" -> "canonical_name"
# All keys should be lowercase for matching

HARDCODED_ALIASES = {
    # === Ukrainian Politicians ===
    # Зеленський
    "зе": "Зеленський",
    "зєля": "Зеленський",
    "зелупа": "Зеленський",
    "зеля": "Зеленський",
    "зелібоба": "Зеленський",
    "слуга": "Зеленський",
    "квартал": "Зеленський",
    "клоун": "Зеленський",
    "президент": "Зеленський",
    "зеленый": "Зеленський",

    # Порошенко
    "порох": "Порошенко",
    "петя": "Порошенко",
    "петро": "Порошенко",
    "барига": "Порошенко",
    "рошен": "Порошенко",
    "шоколадний": "Порошенко",
    "попередник": "Порошенко",

    # Тимошенко
    "юля": "Тимошенко",
    "юлька": "Тимошенко",
    "коса": "Тимошенко",
    "газова принцеса": "Тимошенко",
    "леді ю": "Тимошенко",

    # Янукович
    "янек": "Янукович",
    "овощ": "Янукович",
    "легітимний": "Янукович",
    "межигір'я": "Янукович",

    # Кличко
    "кличко": "Кличко",
    "віталік": "Кличко",
    "боксер": "Кличко",

    # === Russian Politicians ===
    # Путін
    "путін": "Путін",
    "путин": "Путін",
    "пу": "Путін",
    "пуйло": "Путін",
    "хуйло": "Путін",
    "бункерний": "Путін",
    "карлик": "Путін",
    "плішивий": "Путін",
    "вова": "Путін",
    "вовчик": "Путін",

    # Лукашенко
    "лукаш": "Лукашенко",
    "бацька": "Лукашенко",
    "батька": "Лукашенко",
    "картопля": "Лукашенко",
    "таракан": "Лукашенко",
    "усатий": "Лукашенко",

    # === Crypto/Finance ===
    "біток": "біткоін",
    "бітон": "біткоін",
    "btc": "біткоін",
    "bitcoin": "біткоін",
    "біткоїн": "біткоін",

    "ефір": "ефіріум",
    "eth": "ефіріум",
    "ethereum": "ефіріум",
    "етер": "ефіріум",

    "крипта": "криптовалюта",
    "криптовалюти": "криптовалюта",
    "crypto": "криптовалюта",

    "солана": "solana",
    "sol": "solana",

    "додж": "dogecoin",
    "doge": "dogecoin",
    "шиба": "shiba",

    # === Countries/Places ===
    "рашка": "росія",
    "рфія": "росія",
    "московія": "росія",
    "мордор": "росія",

    "піндоси": "америка",
    "сша": "америка",
    "штати": "америка",
    "usa": "америка",

    "гейропа": "європа",
    "єс": "європа",
    "eu": "європа",

    "незалежна": "україна",
    "нєнька": "україна",
    "ua": "україна",

    # === War/Military ===
    "війна": "війна",
    "сво": "війна",
    "спецоперація": "війна",
    "вторгнення": "війна",

    "зсу": "армія",
    "всу": "армія",
    "збройні сили": "армія",

    "орки": "росіяни",
    "рашисти": "росіяни",
    "кацапи": "росіяни",

    # === Tech ===
    "гпт": "chatgpt",
    "чатгпт": "chatgpt",
    "openai": "chatgpt",

    "ілон": "маск",
    "elon": "маск",
    "musk": "маск",
    "тесла": "маск",
    "твіттер": "маск",
}

# Runtime aliases - will be populated by load_aliases()
ENTITY_ALIASES: dict[str, str] = {}
CANONICAL_TO_ALIASES: dict[str, list[str]] = {}

# Database instance (set by init_entity_aliases)
_db_instance = None


def init_entity_aliases(db=None):
    """
    Initialize entity aliases system with database connection.
    Call this at bot startup.
    """
    global _db_instance
    _db_instance = db
    reload_aliases()


def reload_aliases():
    """
    Reload aliases from hardcoded defaults + database.
    Database aliases override hardcoded ones.
    """
    global ENTITY_ALIASES, CANONICAL_TO_ALIASES

    # Start with hardcoded aliases
    ENTITY_ALIASES = dict(HARDCODED_ALIASES)

    # Merge database aliases (they override hardcoded)
    if _db_instance:
        try:
            db_aliases = _db_instance.get_entity_aliases_dict()
            ENTITY_ALIASES.update(db_aliases)
            logger.info(f"Loaded {len(db_aliases)} entity aliases from database")
        except Exception as e:
            logger.warning(f"Failed to load entity aliases from database: {e}")

    # Rebuild reverse mapping
    CANONICAL_TO_ALIASES = {}
    for alias, canonical in ENTITY_ALIASES.items():
        if canonical not in CANONICAL_TO_ALIASES:
            CANONICAL_TO_ALIASES[canonical] = []
        CANONICAL_TO_ALIASES[canonical].append(alias)

    logger.info(f"Total entity aliases: {len(ENTITY_ALIASES)}")


# Initialize with hardcoded aliases by default
reload_aliases()


def expand_aliases(text: str) -> str:
    """
    Expand known aliases in text to include canonical forms.

    Example:
        "що казав про зе?" -> "що казав про зе Зеленський?"
    """
    words = text.lower().split()
    expanded_words = []

    for word in words:
        # Clean punctuation for matching
        clean_word = word.strip('.,!?()[]"\'')

        if clean_word in ENTITY_ALIASES:
            canonical = ENTITY_ALIASES[clean_word]
            # Add both alias and canonical form
            expanded_words.append(word)
            if canonical.lower() not in text.lower():
                expanded_words.append(canonical)
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


def get_canonical(alias: str) -> str:
    """Get canonical form of an alias, or return original if not found."""
    return ENTITY_ALIASES.get(alias.lower(), alias)


def get_all_forms(term: str) -> list[str]:
    """Get all known forms of a term (canonical + aliases)."""
    term_lower = term.lower()

    # Check if it's an alias
    if term_lower in ENTITY_ALIASES:
        canonical = ENTITY_ALIASES[term_lower]
        return [canonical] + CANONICAL_TO_ALIASES.get(canonical, [])

    # Check if it's a canonical form
    if term in CANONICAL_TO_ALIASES:
        return [term] + CANONICAL_TO_ALIASES[term]

    # Check case-insensitive canonical match
    for canonical in CANONICAL_TO_ALIASES:
        if canonical.lower() == term_lower:
            return [canonical] + CANONICAL_TO_ALIASES[canonical]

    return [term]


def expand_query_with_aliases(query: str) -> str:
    """
    Expand a search query with known aliases.
    Adds canonical forms and synonyms for better semantic search.

    Example:
        "порох біткоін" -> "порох Порошенко біткоін bitcoin btc криптовалюта"
    """
    words = query.split()
    expanded = set(words)  # Use set to avoid duplicates

    for word in words:
        clean = word.lower().strip('.,!?()[]"\'')

        # Add canonical form
        if clean in ENTITY_ALIASES:
            canonical = ENTITY_ALIASES[clean]
            expanded.add(canonical)

            # Add a few common aliases for broader search
            aliases = CANONICAL_TO_ALIASES.get(canonical, [])
            for alias in aliases[:3]:  # Limit to avoid query explosion
                expanded.add(alias)

    return " ".join(expanded)
