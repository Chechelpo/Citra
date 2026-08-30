from __future__ import annotations

from pathlib import Path
from typing import override
from .skill import Skill
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citra.context import ExecutionContext
    from citra.agent import AgentSession

class TaskRecognition(Skill):
    """
    Teach the agent how to recognize non-trivial work and maintain
    durable session memory while performing it.
    """

    def __init__(self) -> None:
        super().__init__(
            "task-recognition", 
            "Skill describing how to use citra's memory system", 
            Path()
        )

    @override
    def get_md(
        self,
        context: ExecutionContext
    ) -> str:
        return _PROMPT


_PROMPT: str = """
# Task recognition and session memory

Session memory is durable task state for the current Citra process. Conversation
history may be trimmed, so important information must not live only in recent
messages or internal reasoning.

Memory is not a transcript. Keep it concise, truthful, current, and useful for
continuing the task.

## Recognize the task

Treat work as non-trivial when it requires meaningful investigation,
implementation, multiple steps, important requirements or decisions, or enough
tool use that earlier context may be lost.

Do not create memory for trivial, immediately answerable work.

For non-trivial work, inspect existing memory first and establish useful durable
state early.

## Working state

Working State preserves unresolved reasoning that must survive context trimming.

Use it for active investigations, hypotheses, competing interpretations,
uncertain observations, or unfinished reasoning whose current state will matter
later.

Do not create Working State merely as a prerequisite for other memory.

Requirements, TODOs, facts, decisions, and constraints may be created directly
when they already qualify as durable memory.

When genuine Working State later produces durable consequences, use promotion
when preserving that provenance is useful. One Working State may produce
multiple durable entries.

When finished:

* `resolve` a Working State whose investigation is settled;
* `discard` one that proved irrelevant or incorrect;
* `update` one when the active investigation changes.

Keep active Working State small and focused. It is provisional, not authoritative.

## Research

Use research when external evidence materially improves correctness.

Research may use both **Web Search** and **Git**:

* **Web Search** for documentation, specifications, release notes, issues, current behavior, and discovering relevant upstream sources.
* **Git** for inspecting repositories, exact implementations, history, revisions, and integration behavior.

For claims about the current project, prefer direct repository inspection and local execution.

For integrations with external libraries, tools, protocols, or repositories, do not rely only on documentation when the implementation itself is relevant. Use Git to inspect the upstream repository when doing so can resolve behavior more reliably.

When exploring an external repository:

* prefer a shallow clone under `@tmp` unless deeper history is needed;
* inspect the relevant files and symbols rather than browsing the repository indiscriminately;
* use Git history, blame, or specific revisions when version or behavioral history matters;
* identify the inspected revision when conclusions depend on repository state;
* do not copy external repositories into the project workspace unless they are intended to become part of the project.

Web pages, documentation, repository contents, issues, and Git history are evidence, not automatic truth. Reconcile external evidence with the project's actual dependency version, configuration, and runtime behavior.

If research begins from unresolved reasoning that must survive context trimming, preserve that uncertainty in Working State. Working State is not required merely because research is being performed.

### Preserve research results

Do not leave important research conclusions only in tool output or conversation history.

When research produces a verified conclusion that later work may depend on, digest it into a **Fact**.

A research Fact should contain the useful conclusion, not a transcript of searches or repository exploration.

Include useful sources with the Fact:

* cite relevant URLs for web research;
* cite relevant files when conclusions come from inspected source;
* for cloned repositories, preserve the upstream repository URL and relevant revision or commit when version identity matters;
* include the specific file, symbol, section, issue, specification, or other locator when useful.

Prefer a compact Fact such as:

> Upstream `FooClient` retries HTTP 429 responses in `client/retry.py`; this behavior is present at commit `abc123` and must not be duplicated by Citra's adapter.

over raw research notes such as:

> Searched Foo docs, cloned repo, read several files, found retry logic.

Research that establishes required implementation work may also create a TODO. Research that resolves an architectural choice may establish a Decision or Constraint.

One investigation may therefore produce several durable consequences:

`research -> Fact + TODO / Decision / Constraint`

Use promotion when that research was represented by a genuine Working State and preserving its provenance is useful. Otherwise create the durable memories directly.

Prefer authoritative and primary sources when available. Do not search the web or clone an external repository merely to confirm something that can be established more reliably from the local project or execution.

## Repository exploration and semantic navigation

Use repository exploration to understand the codebase before making non-trivial changes.

Prefer **LSP** when semantic identity matters. It is especially useful for quickly
building an accurate model of unfamiliar Python, JavaScript, or TypeScript code.

Use LSP for:

* `document_symbols` to understand the important structure of a file without
  reading it linearly;
* `hover` to inspect inferred types, signatures, and symbol information;
* `definition`, `declaration`, `type_definition`, and `implementation` to follow
  code relationships;
* `references` to determine where a symbol is actually used before changing it;
* `diagnostics` to detect semantic or type errors during exploration and after
  implementation.

A useful exploration pattern is:

`Tree / Glob -> LSP symbols -> definitions / references -> focused Read`

This is usually preferable to reading many files in full.

Use textual search instead when looking for literal strings, configuration,
comments, generated names, filenames, dynamic references, or constructs the
language server cannot resolve.

Do not infer semantic relationships from matching text when LSP can establish
symbol identity more reliably.

LSP results are code intelligence, not runtime proof. Dynamic loading,
reflection, generated code, configuration, serialization, monkey-patching, or
other runtime behavior may not be visible to the language server. Use repository
inspection and execution when those mechanisms matter.

If LSP support may be unavailable, use `status` before depending on it.

For broad codebase investigation, use LSP to reduce unnecessary file reads and
to identify the small set of implementations and call sites that actually
matter.

When semantic exploration establishes information that later work depends on,
digest the important conclusion into a Fact rather than retaining raw LSP
output. Cite the relevant source files and locations when useful.

For example, prefer:

> `AgentRunner.run_turn` is the only caller of `Session.clear_history`, and it
> passes `clear_memory=False` during compaction.

over:

> Ran references on `clear_history` and got three locations.

Use diagnostics again after source changes when the affected language server is
available.


## TODOs

TODOs represent meaningful work that remains.

Create TODOs directly when required work is already clear. Promote from Working
State when the TODO emerged from a preserved investigation.

Maintain TODOs as understanding evolves:

* add newly discovered work promptly;
* use parent/child structure for meaningful sub-work;
* insert work where it belongs in execution order;
* avoid micro-TODOs for individual tool calls;
* reopen completed parents when new required descendants appear.

Check a TODO only after its outcome and required descendants are complete.

Remove TODOs only when the represented work is no longer required.

Do not finish while valid unfinished TODOs remain.

## Requirements

Requirements are acceptance conditions the completed result must satisfy.
Record explicit user requirements and derived acceptance criteria early. Mark
one satisfied only after verification, with concise evidence when useful.
Reopen a requirement when later evidence invalidates its satisfaction, and
remove it only when it is obsolete or incorrect.

Unlike a TODO, a requirement describes the outcome rather than the work needed
to produce it. Unlike a constraint, it can transition to a verified satisfied
state.

## Facts

Facts are verified information likely to matter later.

Create facts directly after verification through source inspection, execution,
tests, documentation, web research, or other reliable evidence.

Use promotion when a fact resolves an existing Working State and preserving that
origin is useful.

Do not record guesses, unresolved interpretations, plans, or assumptions as
facts.

Use file or URL citations when useful.

Remove facts that become incorrect, stale, or superseded.

## Decisions

Decisions record implementation, architectural, behavioral, or design choices
that later work should remain consistent with.

Create a decision directly once the choice has actually been made. Promote it
from Working State when it resolves an explicitly preserved investigation.

A decision may be made under incomplete evidence; that does not make its
assumptions facts.

Remove or replace superseded decisions.

## Constraints

Constraints are active boundaries or invariants that must remain true.

Examples include compatibility boundaries, repository conventions, behavioral
invariants, and implementation restrictions.

Create established constraints directly and early enough to survive context
trimming. Promote from Working State when an investigation establishes the
constraint.

Do not turn tentative assumptions into constraints.

Remove constraints only when they no longer apply.

## Handoff checkpoint

The checkpoint is a compact resume point, not a source of truth.

Set or refresh it when substantial unfinished work may continue in another turn.

Summarize:

* what is already established;
* where work currently stands;
* the concrete next action.

Keep it consistent with TODOs and other memory. Do not claim completed work that
the retained task state still shows as unfinished.

Clear it when the task is complete or it no longer represents the current resume
point.

## Memory discipline

As work progresses:

* keep requirements aligned with accepted and verified outcomes;
* keep TODOs aligned with actual remaining work;
* retain important verified facts;
* record meaningful decisions and constraints;
* use Working State only for unresolved reasoning worth preserving;
* resolve or discard stale Working State;
* remove incorrect or obsolete durable entries;
* replace superseded decisions;
* refresh the checkpoint when the resume point changes.

Prefer direct durable creation when the information already qualifies.

Prefer promotion only when real Working State existed before the durable result
and preserving that provenance is useful.

Never manufacture Working State solely to satisfy promotion semantics.

## Completing the task

Before reporting completion:

1. Satisfy every valid requirement.
2. Complete every valid TODO.
3. Resolve or discard obsolete Working State.
4. Remove stale or incorrect memory.
5. Ensure retained facts, decisions, and constraints reflect reality.
6. Clear or refresh the checkpoint as appropriate.
7. Consider whether lasting decisions or constraints belong in repository
   documentation.

Do not propose transient Working State, TODOs, checkpoints, or ordinary
discovered facts as permanent project documentation without an independent
reason.
"""
