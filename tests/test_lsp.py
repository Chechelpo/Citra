from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from citra.utils.lsp.client import LspClient
from citra.utils.lsp.capabilities import LspCapabilities
from citra.utils.lsp.config import LspConfig, ServerConfig
from citra.utils.lsp.language import Language, detect_language, server_for_language
from citra.utils.lsp.positions import SourcePosition
from citra.utils.lsp.transport import JsonRpcTransport


def test_javascript_and_typescript_use_typescript_language_server() -> None:
    assert detect_language("app.js") is Language.JAVASCRIPT
    assert detect_language("component.jsx") is Language.JAVASCRIPT
    assert detect_language("app.ts") is Language.TYPESCRIPT
    assert detect_language("component.tsx") is Language.TYPESCRIPT
    assert detect_language("module.mts") is Language.TYPESCRIPT
    assert server_for_language(Language.JAVASCRIPT) == "typescript-language-server"
    assert server_for_language(Language.TYPESCRIPT) == "typescript-language-server"


def test_empty_capability_options_still_mean_supported() -> None:
    capabilities = LspCapabilities.from_server_capabilities(
        {
            "definitionProvider": {},
            "hoverProvider": False,
            "renameProvider": {},
        }
    )
    assert capabilities.definition
    assert capabilities.rename
    assert not capabilities.hover


def test_stdio_framing_and_go_to_definition(tmp_path: Path) -> None:
    source = tmp_path / "app.js"
    source.write_text("function value() { return 1; }\nvalue();\n", encoding="utf-8")
    server_script = Path(__file__).with_name("fake_lsp_server.py")
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    holder = {}
    transport = JsonRpcTransport(
        process,
        notification_handler=lambda method, params: holder["client"].handle_notification(
            method, params
        ),
        request_handler=lambda method, params: holder["client"].handle_request(method, params),
    )
    client = LspClient(
        transport,
        root=tmp_path,
        server=ServerConfig(command=(sys.executable, str(server_script))),
        config=LspConfig(startup_timeout=2, request_timeout=2, diagnostics_timeout=2),
        name="fake-js",
    )
    holder["client"] = client
    try:
        client.initialize()
        uri = client.sync_document(source, source.read_text(encoding="utf-8"), Language.JAVASCRIPT)
        location = client.definitions("definition", uri, SourcePosition(line=1, character=1))
        assert location["uri"] == uri
        assert location["range"]["start"] == {"line": 0, "character": 9}
        assert client.diagnostics(uri) == []
        client.sync_document(
            source,
            source.read_text(encoding="utf-8").replace("return 1", "return 2"),
            Language.JAVASCRIPT,
        )
        assert client.diagnostics(uri) == []
    finally:
        client.close()
        process.wait(timeout=2)
