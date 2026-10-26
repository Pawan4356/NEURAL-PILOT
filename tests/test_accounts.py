import sqlite3

import pytest

from neuralpilot import config
from neuralpilot.accounts import AccountError, AccountStore
from neuralpilot.accounts.passwords import hash_password, verify_password

PASSWORD = "correct-horse-battery"


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.sqlite3")


def test_password_hash_round_trip_and_rejects_wrong_or_malformed_values():
    stored = hash_password(PASSWORD)
    assert stored.startswith("scrypt$")
    assert PASSWORD not in stored
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong-password", stored)
    assert not verify_password(PASSWORD, "not-a-hash")


def test_same_password_gets_a_different_hash_each_time():
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_create_user_normalises_email_and_rejects_duplicates(store):
    user = store.create_user("  Alice@Example.COM ", PASSWORD)
    assert user.email == "alice@example.com"
    with pytest.raises(AccountError, match="already exists"):
        store.create_user("alice@example.com", PASSWORD)


@pytest.mark.parametrize(
    ("email", "password", "message"),
    [
        ("not-an-email", PASSWORD, "valid email"),
        ("a@example.com", "short", "between"),
        ("a@example.com", "x" * 300, "between"),
    ],
)
def test_create_user_rejects_bad_input_with_a_user_message(store, email, password, message):
    with pytest.raises(AccountError, match=message):
        store.create_user(email, password)


def test_authenticate_accepts_only_the_right_password(store):
    user = store.create_user("bob@example.com", PASSWORD)
    assert store.authenticate("BOB@example.com", PASSWORD) == user
    assert store.authenticate("bob@example.com", "wrong-password-here") is None
    assert store.authenticate("nobody@example.com", PASSWORD) is None


def test_session_resolves_to_its_user_until_logout(store):
    user = store.create_user("carol@example.com", PASSWORD)
    token = store.create_session(user.id)
    assert store.user_for_session(token) == user
    store.end_session(token)
    assert store.user_for_session(token) is None


def test_expired_and_unknown_sessions_resolve_to_nobody(store, monkeypatch):
    user = store.create_user("dan@example.com", PASSWORD)
    monkeypatch.setattr(config, "SESSION_TTL_S", -1)
    expired = store.create_session(user.id)
    assert store.user_for_session(expired) is None
    assert store.user_for_session("made-up-token") is None
    assert store.user_for_session(None) is None


def test_session_tokens_are_stored_hashed(store):
    user = store.create_user("erin@example.com", PASSWORD)
    token = store.create_session(user.id)
    rows = sqlite3.connect(store.db_path).execute("SELECT token_hash FROM sessions").fetchall()
    assert rows and token not in rows[0][0]


def test_only_the_owner_can_see_a_run(store):
    alice = store.create_user("alice@example.com", PASSWORD)
    bob = store.create_user("bob@example.com", PASSWORD)
    store.record_run("run-1", alice.id, "churn.csv", "predict churn")

    owned = store.owned_run("run-1", alice.id)
    assert owned.dataset_name == "churn.csv" and owned.goal == "predict churn"
    assert store.owned_run("run-1", bob.id) is None
    assert store.owned_run("missing", alice.id) is None
    assert [run.run_id for run in store.runs_for_user(alice.id)] == ["run-1"]
    assert store.runs_for_user(bob.id) == []
