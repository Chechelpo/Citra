"""CLI inspection for the active workflow and serial role run."""

from __future__ import annotations

from .command import Command, CommandResult


class WorkflowCommand(Command):
    id = "workflow"
    description = "Inspect the workflow and current serial phase."

    def _run(self, args: str) -> CommandResult:
        action = args.strip() or "status"
        if action not in {"status", "show", "cancel"}:
            return CommandResult(output="Usage: /workflow [status|cancel]")

        runtime = self.context.workflow_runtime
        workflow = runtime.workflow
        run = runtime.active_run
        if action == "cancel":
            if not runtime.cancel_run():
                return CommandResult(output="No active workflow run to cancel.")
            return CommandResult(output="Cancelled the active workflow run.")
        sandbox = self.context.sandbox
        lines = [
            f"Workflow: {workflow.name}",
            f"active workflow: {self.context.workflow.name}",
            f"sandbox: {sandbox.mode.name.lower()}",
            "sandbox policy: workflow-owned",
        ]
        if run is None:
            lines.append("run: persistent single-mode session")
            return CommandResult(output="\n".join(lines))

        snapshot = run.snapshot()
        status = (
            "cancelled"
            if snapshot.cancelled
            else "complete"
            if snapshot.completed
            else "running"
        )
        lines.extend(
            (
                f"run: {status}",
                f"phase: {snapshot.current_step or '-'}",
                f"phase executions: {snapshot.execution_count}",
                f"message handoffs: {len(snapshot.handoffs)}",
            )
        )
        if snapshot.handoffs:
            lines.append("transitions:")
            lines.extend(
                f"- {item.step_id} -> {item.next_step}"
                for item in snapshot.handoffs
            )
        return CommandResult(output="\n".join(lines))


__all__ = ["WorkflowCommand"]
