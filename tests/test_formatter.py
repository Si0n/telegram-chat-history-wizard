import pytest
from agent.formatter import Formatter


def test_highlight_single_term():
    f = Formatter()
    escaped = f.escape_html("Я разбираюсь в интеллекте лучше всех")
    result = f.highlight(escaped, ["интеллекте"])
    assert "<b>интеллекте</b>" in result


def test_highlight_case_insensitive():
    f = Formatter()
    escaped = f.escape_html("Привет МИР привет")
    result = f.highlight(escaped, ["мир"])
    assert "<b>МИР</b>" in result


def test_highlight_multiple_terms():
    f = Formatter()
    escaped = f.escape_html("альфа бета гамма")
    result = f.highlight(escaped, ["альфа", "гамма"])
    assert "<b>альфа</b>" in result
    assert "<b>гамма</b>" in result


def test_truncate_html_short():
    f = Formatter()
    result = f.truncate_html("Short text", 100)
    assert result == "Short text"


def test_truncate_html_long():
    f = Formatter()
    result = f.truncate_html("A" * 200, 100)
    assert result.endswith("...")
    # Count visible chars (excluding "...")
    visible = sum(1 for c in result if c != ".")
    assert visible <= 103


def test_truncate_html_preserves_tags():
    f = Formatter()
    text = "Hello <b>world</b> this is a test"
    result = f.truncate_html(text, 11)
    assert "<b>" in result


def test_escape_html():
    f = Formatter()
    result = f.escape_html("2 < 3 & 4 > 1")
    assert "&lt;" in result
    assert "&amp;" in result
    assert "&gt;" in result


def test_format_search_results_empty():
    f = Formatter()
    text, keyboard = f.format_search_results([], 0, 0, [], "asc")
    assert "Nothing found" in text


def test_format_search_results_page():
    f = Formatter()
    results = [
        {"id": 1, "first_name": "Леха", "text": "Тестовое сообщение", "timestamp": "2021-03-15T14:32:00", "chat_id": 100},
        {"id": 2, "first_name": "Саша", "text": "Еще сообщение", "timestamp": "2021-03-15T14:33:00", "chat_id": 100},
        {"id": 3, "first_name": "Дима", "text": "Третье сообщение", "timestamp": "2021-03-15T14:34:00", "chat_id": 100},
    ]
    text, keyboard = f.format_search_results(results, total=3, page=0, highlight_terms=[], sort_order="asc")
    assert "Леха" in text
    assert "Саша" in text
    assert "1-3" in text
