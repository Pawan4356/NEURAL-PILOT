"""Sign-up, login, logout and the `current_user` dependency for run routes.

Login sets an HTTP-only session cookie. The browser sends it back on every
call, and the server looks it up by hash, so the page's scripts never see the
session token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import config
from ..accounts import AccountError, AccountStore, User

SESSION_COOKIE = "np_session"

router = APIRouter(prefix="/api/auth")


def get_accounts() -> AccountStore:
    return AccountStore(config.ACCOUNTS_DB_PATH)


def current_user(request: Request, accounts: AccountStore = Depends(get_accounts)) -> User:
    user = accounts.user_for_session(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(401, "Log in to continue.")
    return user


def _signed_in(accounts: AccountStore, user: User) -> JSONResponse:
    response = JSONResponse({"email": user.email})
    response.set_cookie(
        SESSION_COOKIE,
        accounts.create_session(user.id),
        max_age=config.SESSION_TTL_S,
        httponly=True,
        samesite="lax",
        secure=config.SESSION_COOKIE_SECURE,
    )
    return response


@router.post("/signup")
def sign_up(email: str = Form(...), password: str = Form(...), accounts: AccountStore = Depends(get_accounts)):
    try:
        user = accounts.create_user(email, password)
    except AccountError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _signed_in(accounts, user)


@router.post("/login")
def log_in(email: str = Form(...), password: str = Form(...), accounts: AccountStore = Depends(get_accounts)):
    user = accounts.authenticate(email, password)
    if user is None:
        # Same message for unknown email and wrong password, so accounts cannot be probed.
        raise HTTPException(401, "Email or password is incorrect.")
    return _signed_in(accounts, user)


@router.post("/logout")
def log_out(request: Request, accounts: AccountStore = Depends(get_accounts)) -> JSONResponse:
    accounts.end_session(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/me")
def me(user: User = Depends(current_user)) -> JSONResponse:
    return JSONResponse({"email": user.email})
