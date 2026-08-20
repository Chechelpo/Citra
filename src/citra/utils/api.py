# src/citra/utils/api.py

"""
OpenAI-compatible API helpers shared by the agent loop and commands.
"""

from __future__ import annotations


def chat_completions_url(host: str) -> str:
    """
    Convert the configured OpenAI-compatible API host into its
    Chat Completions endpoint.

    Example:

        https://nanogpt.com/v1
            ->
        https://nanogpt.com/v1/chat/completions
    """

    host = host.rstrip("/")

    if host.endswith("/chat/completions"):
        return host

    return f"{host}/chat/completions"
