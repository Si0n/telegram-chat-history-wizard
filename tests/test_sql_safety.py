import pytest
import tempfile
from pathlib import Path
from db.database import Database


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        return Database(Path(f.name))


def test_select_allowed(db):
    result = db.execute_safe_sql("SELECT 1 AS val")
    assert result[0]["val"] == 1


def test_insert_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("INSERT INTO messages VALUES (1)")


def test_update_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("UPDATE messages SET text='x'")


def test_delete_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("DELETE FROM messages")


def test_drop_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("DROP TABLE messages")


def test_pragma_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("PRAGMA table_info(messages)")


def test_with_cte_allowed(db):
    result = db.execute_safe_sql("WITH cte AS (SELECT 1 AS v) SELECT * FROM cte")
    assert result[0]["v"] == 1


def test_limit_enforced(db):
    result = db.execute_safe_sql("SELECT 1 AS val")
    assert len(result) == 1
    assert result[0]["val"] == 1


def test_lowercase_select_allowed(db):
    result = db.execute_safe_sql("select 1 as val")
    assert result[0]["val"] == 1


def test_leading_whitespace_allowed(db):
    result = db.execute_safe_sql("   SELECT 1 AS val")
    assert result[0]["val"] == 1


def test_semicolon_stripped(db):
    result = db.execute_safe_sql("SELECT 1 AS val;")
    assert result[0]["val"] == 1


def test_trailing_semicolon_stripped(db):
    result = db.execute_safe_sql("SELECT 1 AS val ;  ")
    assert result[0]["val"] == 1
