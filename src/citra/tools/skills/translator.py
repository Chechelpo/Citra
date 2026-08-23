from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

from .skill import Skill

if TYPE_CHECKING:
    from citra.context import ExecutionContext


class Translator(Skill):
    """
    Teach the agent how to debug a web application using Citra's
    subprocess and browser tools.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(
            "media-translator",
            "Reverse-engineer how a game stores, references, assembles, and displays translatable text. Use Python managed with uv to build diagnostic and extraction tools, recover localization context, produce a structured translation corpus, and create a model-agnostic LLM translation runner with deterministic validation.",
            Path(),
        )

    @override
    def get_md(
        self,
        context: ExecutionContext,
    ) -> str:
        return _PROMPT

_PROMPT = """
# Reverse-Engineering Localization Skill

This document details how a translation runner must work for large corpus. Every time this document mentions "visible text" assume its text that the player will see (ex.: dialogue, UI, item names), not source code.

## Objective

The objective of a task of this nature is to provide the following architecture:

`Script extracts visible text from game -> LLM is called for translation -> Translation is validated -> Translation is pasted back into the source game via a script`

## Extracting visible text

This part of the job requires the implementer to:

  - Identify HOW the dialogue is stored
  - Write a script to extract it by trying to mantain in-game scenes.
  - Make sure no extra text is found inside the extraction 

For this reason, this is an iterative task that is focused on writing a script, validating it against the source material and writing the script again.
By "trying to mantain in-game scenes" you'll need to try and, wherever possible, group text together by the actual scene instead of loose extracts, as the llm
needs to know the context.

## LLM for translation

You'll write a chat-completions client that provides:

  A) Host config
  B) Model config
  C) System prompt injection (as append)

This llm will be in charge of translating extracts, but must not be trusted with providing the correct output.

## Back into source
Once you have the translations, you'll need to paste them back into source. This is the riskier task as you'll risk crashing the game with your changes.

For this reason, for testing, you'll take a dataset of translations and do them yourself. You'll test the llm's output against your correct translation. If
anything different in the translated structure is found, adjust the script.

## Your task

First identify if the first two parts have been done already. 

  - **If not:** Your job is to write the script + the agent loop, documenting how you'd expect the extracts to go back into source for the future script that injects them back.
  - **If done:** Your job is to validate extracts and write the back into source script.

Either way, be sure to digest your task into your session memory.
"""