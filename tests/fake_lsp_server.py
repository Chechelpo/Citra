"""Tiny deterministic stdio server used only by transport/client tests."""

from __future__ import annotations

import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\n", b"\r\n"}:
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.casefold()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body)


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "capabilities": {
                        "definitionProvider": True,
                        "hoverProvider": True,
                        "referencesProvider": True,
                        "documentSymbolProvider": True,
                    }
                },
            }
        )
    elif method in {"textDocument/didOpen", "textDocument/didChange"}:
        uri = message["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": []},
            }
        )
    elif method == "textDocument/definition":
        uri = message["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "uri": uri,
                    "range": {
                        "start": {"line": 0, "character": 9},
                        "end": {"line": 0, "character": 14},
                    },
                },
            }
        )
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "exit":
        break
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
