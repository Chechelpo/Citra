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

Session memory is durable working state for the current Citra process.

Conversation history may be trimmed. Do not rely on old messages or old tool
results remaining available when information will matter later.

Memory is not a transcript. Store only state that is useful for continuing,
verifying, or completing the task.

## Recognize the task

At the beginning of a task, determine whether it is trivial or non-trivial.

A trivial task is a small, immediately answerable operation that does not need
a multi-step implementation plan or durable state. Do not create memory merely
to satisfy a procedure for such tasks.

A task is non-trivial when it involves meaningful investigation,
implementation, multiple steps, important requirements, architectural choices,
or work that may span enough tool calls that earlier context could be lost.

For non-trivial tasks, establish useful session memory early.

Before adding new memory, consider any existing TODOs, facts, decisions,
constraints, and checkpoint. Reconcile them with the current request rather
than duplicating or blindly trusting stale state.

## TODOs

Use TODOs as an eager, continuously maintained representation of the work that remains.

For non-trivial tasks, populate the TODO list early with the meaningful work that is already apparent. Do not wait until the entire implementation is understood before recording work, and do not require the initial TODO list to predict every step in advance.

The TODO structure is expected to evolve as investigation and implementation reveal more information.

As you work:

* add newly discovered independent work promptly;
* add hierarchical sub-steps beneath an existing TODO when investigation reveals concrete work required to complete that TODO;
* continue decomposing TODOs into deeper sub-steps when doing so makes the remaining work clearer;
* insert newly discovered work at the appropriate position in the current execution order rather than merely appending everything to the end;
* keep the TODO hierarchy aligned with the actual structure of the work as your understanding improves.

Treat TODO creation as eager planning during execution, not as a one-time planning phase.

When beginning a broad TODO, investigate it and create its necessary sub-steps as soon as they become apparent. Do not keep several known implementation steps only in internal reasoning while leaving the parent TODO vague.

A parent TODO represents the outcome produced by its subtree. Its sub-steps represent concrete work required to reach that outcome.

Do not create unnecessary micro-TODOs for individual tool calls or trivial actions. Decompose work when the resulting sub-steps represent meaningful implementation, investigation, verification, or resolution steps that help track what remains.

Check a TODO only after the work represented by that TODO has actually been completed. A TODO with unfinished descendants is not complete.

Completing all sub-steps does not remove the responsibility to verify and explicitly complete their parent when the parent outcome has been achieved.

If new required work is discovered beneath something previously considered complete, reflect that immediately in the TODO hierarchy and treat the affected work as incomplete again.

Remove a TODO only when it is stale, invalid, irrelevant, redundant, or based on an incorrect assumption. Removing a parent also means the work represented by its subtree is no longer required; do not remove a parent merely to discard still-valid descendants.

Keep completed TODOs when they provide useful execution context, but keep the active hierarchy focused enough that unfinished work remains easy to understand.

The TODO list should answer, at any point during a non-trivial task:

1. What meaningful work is still required?
2. Which larger piece of work does each sub-step belong to?
3. What work has already been completed?
4. What newly discovered work changed or expanded the original plan?
5. What should be worked on next?

Do not allow the TODO state to lag substantially behind the work being performed.

Do not finish a task while valid unfinished TODOs or sub-steps remain.

## Facts

Use facts for important information that has been verified and is likely to
matter in later reasoning.

Good facts include discovered repository behavior, relevant implementation
details, verified command results, important file locations, and other evidence
that future steps may depend on.

When a fact comes from a file or URL, include a useful citation when available.

Do not record guesses, hypotheses, plans, or unresolved interpretations as
facts.

Memory does not replace verification. Re-read files or rerun commands whenever
current state matters.

Remove facts that are discovered to be incorrect, stale, superseded, or no
longer applicable.

## Decisions

Use decisions for implementation, architectural, behavioral, or design choices
that have actually been made.

Do not record a possible option as a decision merely because it is being
considered.

After making a meaningful choice that later implementation must remain
consistent with, record it.

If a decision is reversed or superseded, remove the obsolete decision and
record the replacement when appropriate.

## Constraints

Use constraints for requirements and invariants that must remain true while
performing the task.

Examples include user requirements, compatibility boundaries, repository
conventions, behavioral invariants, and implementation restrictions.

Record important constraints early so they survive context trimming.

Treat retained constraints as active requirements.

Remove a constraint only when evidence shows that it is incorrect, obsolete,
or no longer applicable.

## Handoff checkpoint

Use the checkpoint as a compact resume point for unfinished work.

Set or refresh it after substantial progress when work may continue in another
agent turn, and before ending a turn with unfinished work.

The checkpoint should summarize what is already true and identify the concrete
next action. It should complement TODOs and other memory rather than duplicate
their full contents.

Clear a checkpoint when it no longer represents useful unfinished state, such
as after the task is fully completed.

## Maintain memory while working

Memory must evolve with the task.

When new evidence changes reality:

- add newly discovered required work;
- check work that was actually completed;
- remove invalid TODOs;
- retain important verified facts;
- remove incorrect or obsolete facts;
- record decisions after choices are made;
- replace superseded decisions;
- add newly discovered constraints;
- remove constraints that no longer apply;
- update the handoff checkpoint when the resume point changes.

Keep memory concise, truthful, and current.

Do not keep substantial execution state only in internal reasoning or recent
conversation context.

## Completing the task

Before reporting completion:

1. Ensure every valid TODO is completed.
2. Remove stale or known-incorrect memory entries.
3. Ensure retained facts, decisions, and constraints still reflect reality.
4. Clear or update any obsolete handoff checkpoint.
5. Consider whether important retained decisions or constraints would be useful
   as persistent repository documentation.

Do not propose transient TODOs, checkpoints, or ordinary discovered facts for
project documentation unless there is an independent reason to do so.
"""