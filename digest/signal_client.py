"""Thin client for bbernhard/signal-cli-rest-api."""

from __future__ import annotations

import httpx

from .config import settings


# Signal has no official bot API. We run signal-cli (in the signal-api Docker container), linked to
# the user's Signal account like Signal Desktop, and talk to it over this small local REST API.
def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.signal_api_url, timeout=60)


def list_groups() -> list[dict]:
    with _client() as c:
        r = c.get(f"/v1/groups/{settings.signal_number}")
        r.raise_for_status()
        return r.json()


# Post a message to the configured group, sent as the linked account.
def send(message: str, recipient: str | None = None) -> None:
    recipient = recipient or settings.signal_group_id
    if not (settings.signal_number and recipient):
        raise RuntimeError("SIGNAL_NUMBER and SIGNAL_GROUP_ID must be set to send")
    with _client() as c:
        r = c.post(
            "/v2/send",
            json={
                "message": message,
                "number": settings.signal_number,
                "recipients": [recipient],
                "text_mode": "styled",  # renders **bold** headlines
            },
        )
        r.raise_for_status()
