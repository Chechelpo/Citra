"""Tiny stdio LSP server used by Citra's protocol integration tests."""

from __future__ import annotations

import json
import sys
import time
from typing import Any


MODE = sys.argv[1] if len(sys.argv) > 1 else "basic"
config_ok = False
pending_config_uri: tuple[str, int] | None = None
pull_count = 0


def send(message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\n", b"\r\n"}:
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.casefold().strip()] = value.strip()
    length = int(headers["content-length"])
    value = json.loads(sys.stdin.buffer.read(length).decode("utf-8"))
    return value if isinstance(value, dict) else None


def publish(uri: str, version: int, message: str = "fake diagnostic") -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "version": version,
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                        "severity": 1,
                        "source": "fake",
                        "message": message,
                    }
                ],
            },
        }
    )


while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params")

    if method == "initialize" and request_id is not None:
        capabilities: dict[str, Any] = {
            "textDocumentSync": {"openClose": True, "change": 2},
            "definitionProvider": True,
        }
        if MODE == "pull":
            capabilities["diagnosticProvider"] = {"interFileDependencies": False, "workspaceDiagnostics": False}
        send({"jsonrpc": "2.0", "id": request_id, "result": {"capabilities": capabilities}})
        continue

    if method == "initialized":
        if MODE == "configuration":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 900,
                    "method": "workspace/configuration",
                    "params": {"items": [{"section": "python"}, {"section": "pyright"}]},
                }
            )
        elif MODE in {"dynamic", "dynamic-pull-delayed", "dynamic-replace"}:
            if MODE == "dynamic-pull-delayed":
                # Reproduce modern Pyright's timing: initialize says nothing
                # about diagnosticProvider, but pull diagnostics are registered
                # asynchronously after ``initialized``.
                time.sleep(0.10)
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 901,
                    "method": "client/registerCapability",
                    "params": {
                        "registrations": [
                            {
                                "id": "diag-old" if MODE == "dynamic-replace" else "diag",
                                "method": "textDocument/diagnostic",
                                "registerOptions": {
                                    "identifier": "fake-old" if MODE == "dynamic-replace" else "fake-dynamic"
                                },
                            }
                        ]
                    },
                }
            )
        continue

    if request_id == 900 and method is None:
        result = message.get("result")
        config_ok = (
            isinstance(result, list)
            and len(result) == 2
            and result[0] == {"analysis": {"diagnosticMode": "openFilesOnly"}}
            and result[1] == {}
        )
        if config_ok and pending_config_uri is not None:
            publish(*pending_config_uri, message="configuration ok")
            pending_config_uri = None
        continue

    if MODE == "dynamic-replace" and request_id == 901 and method is None:
        send(
            {
                "jsonrpc": "2.0",
                "id": 902,
                "method": "client/registerCapability",
                "params": {
                    "registrations": [
                        {
                            "id": "diag-new",
                            "method": "textDocument/diagnostic",
                            "registerOptions": {"identifier": "fake-new"},
                        }
                    ]
                },
            }
        )
        continue

    if MODE == "dynamic-replace" and request_id == 902 and method is None:
        # Match Pyright DynamicFeature.register(): replacement registration is
        # established first, then the old Disposable unregisters its id.
        send(
            {
                "jsonrpc": "2.0",
                "id": 903,
                "method": "client/unregisterCapability",
                "params": {
                    "unregistrations": [
                        {"id": "diag-old", "method": "textDocument/diagnostic"}
                    ]
                },
            }
        )
        continue

    if MODE == "dynamic-replace" and request_id == 903 and method is None:
        continue

    if method == "textDocument/didOpen" and isinstance(params, dict):
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        version = doc.get("version", 1)
        if not isinstance(uri, str) or not isinstance(version, int):
            continue
        if MODE == "exit-on-open":
            sys.exit(23)
        if MODE == "configuration":
            if config_ok:
                publish(uri, version, "configuration ok")
            else:
                pending_config_uri = (uri, version)
        elif MODE in {"push", "cold-only"}:
            time.sleep(0.15)
            publish(uri, version, "cold push")
        elif MODE == "basic":
            send({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "version": version, "diagnostics": []},
            })
        elif MODE == "stale":
            publish(uri, version, "initial")
        continue

    if method == "textDocument/didChange" and isinstance(params, dict):
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        version = doc.get("version", 1)
        if MODE == "push" and isinstance(uri, str) and isinstance(version, int):
            publish(uri, version, "warm push")
        elif MODE == "basic" and isinstance(uri, str) and isinstance(version, int):
            send({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "version": version, "diagnostics": []},
            })
        elif MODE == "stale" and isinstance(uri, str) and isinstance(version, int):
            publish(uri, version - 1, "stale")
            publish(uri, version, "current")
        continue

    if method == "textDocument/definition" and request_id is not None:
        uri = params.get("textDocument", {}).get("uri") if isinstance(params, dict) else None
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "uri": uri,
                "range": {
                    "start": {"line": 0, "character": 9},
                    "end": {"line": 0, "character": 14},
                },
            },
        })
        continue

    if method == "textDocument/diagnostic" and request_id is not None:
        if MODE != "pull":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"kind": "full", "items": []}})
            continue
        pull_count += 1
        previous_result_id = params.get("previousResultId") if isinstance(params, dict) else None
        if previous_result_id == "r1":
            result = {"kind": "unchanged", "resultId": "r1"}
        else:
            result = {
                "kind": "full",
                "resultId": "r1",
                "items": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                        "severity": 1,
                        "source": "fake-pull",
                        "message": "pulled diagnostic",
                    }
                ],
            }
        send({"jsonrpc": "2.0", "id": request_id, "result": result})
        continue

    if method == "shutdown" and request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
        continue
    if method == "exit":
        break

    if request_id is not None and method is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
