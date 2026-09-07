"""Contract tests for the model-facing Python environment tool."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from citra.context.workspace_context.runtime import (
    ProvisionedTool,
    RuntimeProvisioning,
)
from citra.tools.capabilities import ToolCapabilities
from citra.tools.default_registry import (
    ToolSet,
    all_tools,
)
from citra.tools.transient.python import (
    Python,
    _parse_package_list,
    _require_version,
)
from citra.workflows import (
    ArchitectWorkflow,
    ChatWorkflow,
    TaskWorkflow,
)


class _RecordedRun:
    """Capture the commands handed to a fake sandbox."""

    def __init__(self, output: str = "", returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode
        self.timed_out = False
        self.commands: list[list[str]] = []
        self.last_timeout: int = 0
        self.last_network: bool | None = None
        self.network_requests: list[bool] = []

    @property
    def last_command(self) -> list[str]:
        """Return the most recent command, or an empty list when none."""
        return self.commands[-1] if self.commands else []


class _FakeSandbox:
    """Minimal sandbox that records calls and returns canned output."""

    def __init__(self, output: str = "", returncode: int = 0) -> None:
        self._output = output
        self._returncode = returncode
        self.record = _RecordedRun(output, returncode)

    def run(
        self,
        command,
        *,
        cwd,
        timeout,
        network,
        environment=None,
    ) -> _RecordedRun:
        del cwd, environment
        self.record.commands.append(list(command))
        self.record.last_timeout = timeout
        self.record.last_network = network
        self.record.network_requests.append(network)
        self.record.output = self._output
        self.record.returncode = self._returncode
        self.record.timed_out = False
        return self.record


class _FakeWorkspace:
    """Stand-in exposing the Python tool's required workspace surface."""

    def __init__(self, root: Path) -> None:
        self.workspace = root
        self._registered: list[str] = []
        self._python = root / "env" / "python" / "bin" / "python"
        self._python.parent.mkdir(parents=True, exist_ok=True)
        self._python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def python_runtime(self) -> Path:
        return self.workspace / "env" / "python"

    def python_bin(self) -> Path:
        return self.workspace / "env" / "python" / "bin"

    def resolve_path(self, raw: str | Path) -> Path:
        return (self.workspace / str(raw)).resolve()

    def display_path(self, raw: str | Path) -> str:
        return str(raw)

    def refresh_staged_command(self, command: str):
        self._registered.append(command)
        return self.python_bin() / command

    @property
    def registered_commands(self) -> tuple[str, ...]:
        return tuple(self._registered)


def _context(
    workspace_root: Path,
    *,
    output: str = "",
    returncode: int = 0,
):
    """Build the minimum context the Python tool needs."""
    workspace = _FakeWorkspace(workspace_root)
    sandbox = _FakeSandbox(output=output, returncode=returncode)
    config = SimpleNamespace(
        model=lambda: SimpleNamespace(id="test-model"),
        python=SimpleNamespace(
            always_allow_network=True,
            permission_timeout=30,
        ),
    )
    return SimpleNamespace(
        workspace=workspace,
        sandbox=sandbox,
        config=config,
        model_config=lambda: SimpleNamespace(id="test-model"),
    )


class PythonToolRegistryTests(unittest.TestCase):
    """The Python tool is registered as a deferred tool by default."""

    def test_python_is_a_deferred_tool_in_default_registry(self) -> None:
        deferred = all_tools(are_deferred=True)
        core = all_tools(are_deferred=False)
        self.assertIn(Python, deferred)
        self.assertNotIn(Python, core)

    def test_chat_workflow_exposes_python_as_deferred(self) -> None:
        self.assertIn(Python, ChatWorkflow._TOOLS.deferred_tools)

    def test_task_workflow_exposes_python_as_deferred(self) -> None:
        self.assertIn(Python, TaskWorkflow._TOOLS.deferred_tools)

    def test_architect_workflow_exposes_python_via_default_registry(
        self,
    ) -> None:
        self.assertIn(Python, ArchitectWorkflow._TOOLS.deferred_tools)

    def test_python_toolset_advertises_full_action_set(self) -> None:
        toolset = ToolSet(core_tools=(), deferred_tools=(Python,))
        configuration = toolset.get_configuration_with_id(Python.TOOL_ID)
        self.assertIsNotNone(configuration)
        self.assertEqual(
            configuration.capabilities.enabled_actions,
            Python.ACTIONS,
        )


class PythonToolSchemaTests(unittest.TestCase):
    """The schema and capability declarations are model-facing correct."""

    def test_tool_id_is_python(self) -> None:
        self.assertEqual(Python.TOOL_ID, "python")

    def test_action_enum_matches_capabilities(self) -> None:
        definition = Python.CITRA_DEFINITION
        action_property = next(
            prop
            for prop in definition.function.parameters.properties
            if prop.name == "action"
        )
        self.assertEqual(tuple(action_property.schema.enum), Python.ACTIONS)
        self.assertEqual(
            action_property.schema.enum,
            Python.CAPABILITIES.actions,
        )

    def test_capability_restriction_filters_action_enum(self) -> None:
        restricted = ToolCapabilities(include=("status", "sync"))
        bound = Python.configure_capabilities(restricted)
        # enabled_actions preserve the tool's declared action order, not
        # the inclusion declaration order.
        self.assertEqual(bound.enabled_actions, ("sync", "status"))
        definition = bound.apply_to_definition(Python.CITRA_DEFINITION)
        action_property = next(
            prop
            for prop in definition.function.parameters.properties
            if prop.name == "action"
        )
        self.assertEqual(action_property.schema.enum, ("sync", "status"))


class PythonVersionValidationTests(unittest.TestCase):
    """`_require_version` accepts the surface uv understands and rejects junk."""

    def test_accepts_dotted_patch(self) -> None:
        self.assertEqual(_require_version("3.12"), "3.12")
        self.assertEqual(_require_version("3.12.3"), "3.12.3")

    def test_accepts_implementation_qualified(self) -> None:
        self.assertEqual(_require_version("cpython-3.12.3"), "cpython-3.12.3")
        self.assertEqual(_require_version("pypy-3.10"), "pypy-3.10")
        self.assertEqual(_require_version("graalpy-3.12.0"), "graalpy-3.12.0")

    def test_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(_require_version("  3.12  "), "3.12")

    def test_rejects_blank_or_non_string(self) -> None:
        for bad in ("", "   ", None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                _require_version(str(bad) if bad is not None else "")

    def test_rejects_garbage(self) -> None:
        for bad in ("foo", "foo bar", "3.x", "3.12-rc1", "cpython", "3."):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                _require_version(bad)


class PythonPackageListTests(unittest.TestCase):
    """`_parse_package_list` deduplicates and rejects unsupported shapes."""

    def test_string_is_treated_as_single_requirement(self) -> None:
        self.assertEqual(
            _parse_package_list("numpy", field_name="packages"),
            ("numpy",),
        )

    def test_list_is_deduplicated(self) -> None:
        self.assertEqual(
            _parse_package_list(
                ["numpy", "scikit-learn", "numpy"],
                field_name="packages",
            ),
            ("numpy", "scikit-learn"),
        )

    def test_blank_entries_are_dropped(self) -> None:
        self.assertEqual(
            _parse_package_list(["  ", "numpy", ""], field_name="packages"),
            ("numpy",),
        )

    def test_non_string_non_list_is_type_error(self) -> None:
        with self.assertRaises(TypeError):
            _parse_package_list(123, field_name="packages")

    def test_too_many_entries_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_package_list(
                [f"pkg{i}" for i in range(65)],
                field_name="packages",
            )


class PythonToolExecutionTests(unittest.TestCase):
    """`_execute` dispatches to the right uv subcommand and venv path."""

    def setUp(self) -> None:
        self.temporary = __import__("tempfile").TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.context = _context(self.root, output="ok", returncode=0)
        # Bypass Tool.__init__ to skip the schema/capability validation that
        # would require a real ExecutionContext. ``_execute`` only reads
        # ``self.context``, so the test can call it directly.
        self.tool = Python.__new__(Python)
        object.__setattr__(self.tool, "_Tool__context", self.context)
        object.__setattr__(
            self.tool,
            "_Tool__capabilities",
            Python.CAPABILITIES,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_status_reports_venv_and_interpreter(self) -> None:
        result = self.tool._execute({"action": "status"})
        self.assertIn("venv:", result)
        self.assertIn("interpreter:", result)
        self.assertIn("ok", result)
        # status calls both ``python --version`` and ``uv pip list``.
        commands = self.context.sandbox.record.commands
        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0][0].endswith("/env/python/bin/python"))
        self.assertEqual(commands[0][1:], ["--version"])
        self.assertEqual(commands[1][0:3], ["uv", "pip", "list"])

    def test_select_version_invokes_uv_python_install_then_uv_venv(self) -> None:
        result = self.tool._execute(
            {
                "action": "select_version",
                "version": "3.12",
                "reason": "Download the requested Python runtime.",
            }
        )
        self.assertIn("Selected Python version: 3.12", result)
        # The second run was uv venv --seed --python 3.12 <env_root>
        self.assertEqual(
            self.context.sandbox.record.last_command[0:3],
            ["uv", "venv", "--seed"],
        )
        # After select_version we re-register the managed python/python3.
        self.assertIn("python", self.context.workspace.registered_commands)
        self.assertIn("python3", self.context.workspace.registered_commands)
        self.assertTrue(self.context.sandbox.record.last_network)

    def test_select_version_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            self.tool._execute({"action": "select_version", "version": "foo"})

    def test_install_passes_packages_to_uv_pip_install(self) -> None:
        result = self.tool._execute(
            {
                "action": "install",
                "packages": ["numpy", "scikit-learn"],
                "reason": "Install the requested project dependencies.",
            }
        )
        self.assertIn("Installed 2 package(s)", result)
        self.assertEqual(
            self.context.sandbox.record.last_command[0:3],
            ["uv", "pip", "install"],
        )
        self.assertEqual(
            self.context.sandbox.record.last_command[3:],
            ["numpy", "scikit-learn"],
        )
        self.assertIn("python", self.context.workspace.registered_commands)
        self.assertTrue(self.context.sandbox.record.last_network)

    def test_install_supports_dev_extra(self) -> None:
        self.tool._execute(
            {
                "action": "install",
                "packages": ["pytest"],
                "dev": True,
                "reason": "Install the requested development dependency.",
            }
        )
        self.assertEqual(
            self.context.sandbox.record.last_command[3:6],
            ["--extra", "dev", "pytest"],
        )

    def test_install_requires_packages(self) -> None:
        with self.assertRaises(ValueError):
            self.tool._execute({"action": "install"})

    def test_uninstall_uses_uv_pip_uninstall(self) -> None:
        self.tool._execute({"action": "uninstall", "packages": ["numpy"]})
        self.assertEqual(
            self.context.sandbox.record.last_command[0:3],
            ["uv", "pip", "uninstall"],
        )
        self.assertEqual(
            self.context.sandbox.record.last_command[3:],
            ["numpy"],
        )
        self.assertFalse(self.context.sandbox.record.last_network)

    def test_sync_targets_project_pyproject(self) -> None:
        project_pyproject = self.root / "pyproject.toml"
        project_pyproject.write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        self.tool._execute(
            {
                "action": "sync",
                "reason": "Download dependencies declared by the project.",
            }
        )
        self.assertEqual(
            self.context.sandbox.record.last_command[0:5],
            ["uv", "sync", "--active", "--project", str(self.root)],
        )
        self.assertTrue(self.context.sandbox.record.last_network)

    def test_sync_requires_pyproject(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.tool._execute(
                {"action": "sync", "reason": "Sync project dependencies."}
            )

    def test_failed_command_returns_failure_marker(self) -> None:
        self.context.sandbox = _FakeSandbox(output="oh no", returncode=1)
        result = self.tool._execute(
            {
                "action": "install",
                "packages": ["broken"],
                "reason": "Install the requested dependency.",
            }
        )
        # The Python tool renders failures as a string; it does not raise.
        self.assertIn("(exit code 1)", result)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.tool._execute({"action": "garbage"})

    def test_timeout_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            self.tool._execute(
                {
                    "action": "install",
                    "packages": ["numpy"],
                    "reason": "Install the requested dependency.",
                    "timeout": 1,
                }
            )

    def test_network_backed_actions_require_a_reason(self) -> None:
        for arguments in (
            {"action": "select_version", "version": "3.12"},
            {"action": "install", "packages": ["numpy"]},
        ):
            with self.subTest(action=arguments["action"]), self.assertRaisesRegex(
                ValueError, "reason"
            ):
                self.tool._execute(arguments)

    def test_network_denial_prevents_uv_execution(self) -> None:
        self.context.config.python.always_allow_network = False
        with unittest.mock.patch(
            "citra.tools.transient.python.PromptUser._execute",
            return_value="Deny",
        ):
            result = self.tool._execute(
                {
                    "action": "install",
                    "packages": ["numpy"],
                    "reason": "Install the requested dependency.",
                }
            )
        self.assertIn("permission-denied", result)
        self.assertEqual(self.context.sandbox.record.commands, [])

    def test_network_approval_uses_configured_timeout(self) -> None:
        self.context.config.python.always_allow_network = False
        self.context.config.python.permission_timeout = 47
        with unittest.mock.patch(
            "citra.tools.transient.python.PromptUser._execute",
            return_value="Allow once",
        ) as prompt:
            self.tool._execute(
                {
                    "action": "install",
                    "packages": ["numpy"],
                    "reason": "Install the requested dependency.",
                }
            )
        self.assertEqual(prompt.call_args.args[0]["timeout"], 47)
        self.assertTrue(self.context.sandbox.record.last_network)

    def test_status_does_not_request_network(self) -> None:
        self.tool._execute({"action": "status"})
        self.assertEqual(
            self.context.sandbox.record.network_requests,
            [False, False],
        )


class PythonRuntimeResolutionTests(unittest.TestCase):
    """Staged ``python`` commands win over host-discovered ones."""

    def _provisioning(
        self,
        *,
        has_staged: bool = True,
    ) -> RuntimeProvisioning:
        tools: dict[str, ProvisionedTool] = {
            "command:python": ProvisionedTool(
                id="command:python",
                commands={"python": Path("/host/usr/bin/python")},
                mode="host",
            ),
            "command:python3": ProvisionedTool(
                id="command:python3",
                commands={"python3": Path("/host/usr/bin/python3")},
                mode="host",
            ),
        }
        if has_staged:
            tools["staged:python"] = ProvisionedTool(
                id="staged:python",
                commands={
                    "python": Path("/agent/env/python/bin/python"),
                },
                mode="dependency-environment",
                health="not-configured",
            )
        return RuntimeProvisioning(
            runtime_root=Path("/runtime"),
            budget_bytes=0,
            copied_bytes=0,
            assets={},
            tools=tools,
            definitions={},
        )

    def test_staged_python_wins_over_host(self) -> None:
        resolved = self._provisioning().resolve_command("python")
        self.assertEqual(str(resolved), "/agent/env/python/bin/python")

    def test_unstaged_command_falls_back_to_host(self) -> None:
        resolved = self._provisioning(has_staged=False).resolve_command("python")
        self.assertEqual(str(resolved), "/host/usr/bin/python")

    def test_register_staged_command_redirects_subsequent_lookups(self) -> None:
        provisioning = self._provisioning(has_staged=False)
        # Before registering, the host python is returned.
        self.assertEqual(
            str(provisioning.resolve_command("python")),
            "/host/usr/bin/python",
        )
        provisioning.register_staged_command(
            "python", Path("/agent/env/python/bin/python")
        )
        self.assertEqual(
            str(provisioning.resolve_command("python")),
            "/agent/env/python/bin/python",
        )


if __name__ == "__main__":
    unittest.main()
