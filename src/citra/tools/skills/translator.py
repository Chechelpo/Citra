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

Build a localization pipeline for large media corpora, primarily games.

"Visible text" means text shown to the player: dialogue, UI, descriptions, item names, etc. Source code is never considered visible text.

## Objective

Produce this pipeline:

`Extract visible text -> Build context -> LLM translates -> Validate deterministically -> Reinsert into source`

The LLM is a translator, not a source-of-truth database. Extraction, identity, validation, and reinsertion must remain deterministic whenever possible.

## Translation Rules

Do not translate:

* Source code.
* Existing Latin-script character names unless explicitly required by the project.
* Protected tokens, placeholders, tags, control codes, or formatting markers.

Names written in the source language may be transliterated or localized according to the project's naming policy.

Never infer that two characters are the same because their names are identical or similar.

## Extraction

Reverse-engineer how visible text is stored, referenced, assembled, and displayed.

Build an extraction script that:

* Extracts only visible text.
* Preserves stable source references for reinsertion.
* Preserves dialogue order.
* Groups dialogue into scenes whenever possible.
* Preserves source IDs for speakers and other entities.

Extraction is iterative: inspect source data, extract, compare against the game, and refine until false positives and missing text are minimized.

Each translation unit should have a stable ID and enough metadata to locate its exact source field later.

## Translation Runner

Build a configurable chat-completions client with:

* Host configuration.
* Model configuration.
* Appended system-prompt configuration.
* Structured output.
* Tool calling.
* Retries for 429 and recoverable 5xx errors.

Never trust LLM output without validation.

## Context Tools

Keep persistent context outside the model.

### Entities

Entity IDs are authoritative. Names are labels, not identities.

Provide operations equivalent to:

* `entity.get(id)`
* `entity.search(query)`
* `entity.add_note(id, note, certainty)`
* `entity.add_alias(id, alias)`

Never merge, split, or identify entities based only on names.

Useful entity context includes aliases, descriptions, speech patterns, relationships, and supporting source references.

### Scenes

Provide:

* `scene.get(id)`
* `scene.search(query)`
* `scene.add_note(id, note, certainty)`

Scenes should preserve participants, ordering, location, surrounding scenes, source references, and concise factual summaries.

### Translation Memory

Provide:

* `translation.get(unit_id)`
* `translation.search(text, category=None, scope=None)`
* `translation.save(unit_id, translation, metadata)`

Matching source strings do not necessarily require matching translations.

### Glossary

Provide:

* `glossary.lookup(term, scope=None)`
* `glossary.set(term, translation, scope, constraints=None)`

Glossary rules may be global or scoped to a scene, character, relationship, item family, gameplay system, or UI domain. More specific rules override broader ones.

### Source Context

Provide:

* `source.get_context(unit_id, before, after)`
* `source.inspect_reference(unit_id)`

These expose surrounding visible text and source metadata without treating nearby code as translatable content.

### Context Builder

Provide:

* `context.build(unit_ids)`

It should gather only relevant media information, scene context, entities, nearby dialogue, glossary rules, translation memory, and source metadata.

The runner should build context automatically rather than forcing the LLM to rediscover it through repeated tool calls.

### Web Research

Integrate the available SearXNG search capability into the translation runner as an optional research tool.

Expose it to the translation model for cases where external context may improve translation, such as:

* Proper nouns and official localized names.
* Franchise-specific terminology.
* Idioms, slang, and wordplay.
* Cultural, historical, religious, or technical references.
* Existing localization conventions.

The runner should instruct the translator to prefer targeted searches using the original-language term and relevant media context.

Search results are supporting evidence only. The translation prompt must make clear that web results cannot override extracted entity IDs, source metadata, scene structure, glossary locks, or other authoritative project data.

Prefer official and primary sources where identifiable, while treating wikis, forums, fan translations, and machine translations as secondary evidence.

Avoid unnecessary searches when equivalent context is already available through the corpus, entities, scenes, glossary, or translation memory.

## Tool Rules

Tool results are authoritative only for fields explicitly returned.

The model must not:

* Infer entity equivalence.
* Invent canonical IDs.
* Modify source identity.
* Turn uncertain observations into confirmed facts.

Model-discovered information should include confidence and supporting source units where possible.

## Translation Prompt

Each request must contain:

Permanent context:

1. Role: `You are a <source language> to English translator translating <media name>.`
2. Concise information about the media relevant to translation.
3. Identity and formatting rules.

Current context:

1. Translation category: dialogue, item, description, UI, name, etc.
2. Applicable entities, scene context, terminology, and nearby text.
3. Strictness:

   * Names, labels, UI, and terminology: prefer direct and consistent translation.
   * Dialogue: preserve meaning, characterization, tone, and natural English.

Require structured output and nothing outside that schema.

The model may report uncertainty instead of guessing.

Useful flags include:

* `ambiguous_speaker`
* `ambiguous_referent`
* `missing_context`
* `terminology_conflict`
* `wordplay`
* `uncertain_proper_noun`
* `incomplete_source`

## Validation

Validate every model response deterministically.

At minimum verify:

* Every requested unit ID appears exactly once.
* No unknown units were added.
* The output schema is valid.
* Protected tokens and formatting are preserved.
* Required untranslated text is unchanged.
* Category-specific constraints are satisfied.
* The result can be serialized safely back into the source format.

The LLM must never be the sole validator of LLM output.

## Reinsertion

Reinsert translations using the source references preserved during extraction.

This stage must avoid changing unrelated source structure.

Before translating the full corpus:

1. Create a small manually translated fixture set.
2. Run it through extraction, translation parsing, validation, and reinsertion.
3. Compare the produced structure against the expected structure.
4. Fix the pipeline until differences are understood and safe.

## Task Selection

First inspect the existing project.

If extraction and the translation runner are incomplete:

* Build them.
* Document the source metadata required for future reinsertion.

If both already exist:

* Validate the extracted corpus.
* Build and test the reinsertion script.

Preserve important discoveries in the project's persistent context as you work.
"""