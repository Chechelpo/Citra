# Task Set: Standalone Multi-Language LSP Subsystem

## Objective

Implement a production-quality Language Server Protocol subsystem targeting the LSP 3.17 feature model.

The subsystem must be self-contained inside a single `lsp/` package and must not depend on the architecture of the application embedding it.

Initial supported languages:

```text
TypeScript
JavaScript
Python
Java
HTML
Vue
CSS
```

Default language servers:

| Language   | Server                       | Default command                       |
| ---------- | ---------------------------- | ------------------------------------- |
| TypeScript | TypeScript Language Server   | `typescript-language-server --stdio`  |
| JavaScript | TypeScript Language Server   | `typescript-language-server --stdio`  |
| Python     | Pyright                      | `pyright-langserver --stdio`          |
| Java       | Eclipse JDT Language Server  | `jdtls`                               |
| HTML       | VS Code HTML Language Server | `vscode-html-language-server --stdio` |
| CSS        | VS Code CSS Language Server  | `vscode-css-language-server --stdio`  |
| Vue        | Vue Language Server          | `vue-language-server --stdio`         |

Do not automatically install or download language servers.

Missing servers must produce explicit availability information.

---

# 1. Package boundary

Everything required to communicate with and manage language servers should live under one package:

```text
lsp/
├── __init__.py
├── manager.py
├── client.py
├── transport.py
├── protocol.py
├── capabilities.py
├── language.py
├── project_root.py
├── positions.py
├── documents.py
├── diagnostics.py
├── discovery.py
├── workspace_edit.py
├── formatting.py
├── config.py
├── errors.py
└── servers/
    ├── __init__.py
    ├── base.py
    ├── typescript.py
    ├── pyright.py
    ├── jdtls.py
    ├── html.py
    ├── css.py
    └── vue.py
```

The exact decomposition may change, but these responsibilities must remain separated.

The subsystem must not require knowledge of:

```text
model tools
agent loops
terminal UIs
conversation state
application registries
prompt construction
```

The host application should interact with the subsystem only through exported LSP primitives.

---

# 2. Core architecture

The intended lifetime model is:

```text
Host application
│
└── LspManager
    │
    ├── LspClient(root=A, server=pyright)
    │   └── persistent server process
    │
    ├── LspClient(root=B, server=typescript)
    │   └── persistent server process
    │
    └── LspClient(root=B, server=vue)
        └── persistent server process
```

`LspManager` owns server-process lifetime.

`LspClient` owns one initialized LSP connection.

`LspTransport` owns JSON-RPC framing and request correlation.

Language-server processes must be persistent relative to the manager lifetime.

Repeated semantic operations must reuse an existing healthy server when they resolve to the same:

```text
project root
+
server identity
```

Never start and stop a language server for each request.

Never start one language server independently for every file in a recursive operation.

---

# 3. Public interaction primitives

The package must expose enough high-level primitives that callers do not need to construct raw LSP JSON.

A reasonable top-level API is:

```python
manager = LspManager(config)

client = manager.get_client(path)

client.sync_document(path)

client.diagnostics(path)
client.hover(path, position)
client.go_to(path, position, kind)
client.references(path, position)
client.document_symbols(path)
client.workspace_symbols(query)
client.completion(path, position)
client.signature_help(path, position)
client.code_actions(path, source_range)
client.rename(path, position, new_name)
client.format(path)
client.format_range(path, source_range)
client.call_hierarchy(path, position, direction)
client.type_hierarchy(path, position, direction)

manager.diagnostics_tree(root)
manager.status()
manager.shutdown_workspace(root)
manager.shutdown_all()
```

The lower-level client must also expose generic protocol primitives where necessary:

```python
client.request(method, params, timeout=None)

client.notify(method, params)

client.sync_document(path)
```

The transport layer must expose:

```python
transport.request(method, params, timeout)

transport.notify(method, params)

transport.respond(request_id, result)

transport.respond_error(
    request_id,
    code,
    message,
    data=None,
)
```

Callers should normally use semantic `LspClient` methods rather than `request()` directly.

---

# 4. `LspManager`

Implement:

```python
class LspManager:
    ...
```

The manager owns active clients.

Clients must be keyed by:

```python
@dataclass(frozen=True)
class ClientKey:
    root: Path
    server_id: str
```

Provide equivalents of:

```python
get_client(path)

get_client_for_language(
    language,
    root,
)

status()

shutdown_workspace(workspace)

shutdown_all()
```

The manager must:

* lazily start servers;
* reuse initialized clients;
* prevent duplicate startup races;
* detect dead processes;
* track process PID;
* expose lifecycle state;
* expose capabilities;
* isolate unrelated project roots;
* cache server-unavailable states;
* support controlled crash recovery;
* prevent rapid restart loops;
* shut down clients cleanly.

The manager itself should be long-lived.

---

# 5. Server definitions

Server-specific behavior belongs in `servers/`.

Define a reusable server contract such as:

```python
class LanguageServerDefinition(ABC):
    id: str
    languages: tuple[Language, ...]
    default_command: tuple[str, ...]

    def supports_path(
        self,
        path: Path,
    ) -> bool:
        ...

    def project_root(
        self,
        workspace: Path,
        path: Path,
    ) -> Path:
        ...

    def initialization_options(
        self,
        root: Path,
    ) -> dict[str, Any] | None:
        ...

    def settings(
        self,
        root: Path,
    ) -> dict[str, Any]:
        ...
```

Server adapters contain only server-specific policy.

They must not reimplement:

```text
JSON-RPC framing
request correlation
document state
diagnostic storage
generic LSP requests
process reuse
```

---

# 6. Transport

Implement correct LSP stdio framing:

```text
Content-Length: <number>\r\n
\r\n
<JSON body>
```

The transport must support:

```python
request(method, params, timeout)

notify(method, params)

respond(request_id, result)

respond_error(request_id, code, message, data=None)
```

Incoming messages include:

```text
responses
notifications
server -> client requests
```

Use one continuously running stdout reader.

Do not create a reader thread per request.

Continuously drain stderr independently.

Writes to stdin must be serialized so JSON-RPC messages can never interleave.

---

# 7. Request correlation

Use monotonically increasing JSON-RPC request IDs.

Maintain:

```text
request_id -> pending request
```

A pending request must support:

```text
result
JSON-RPC error
timeout
server termination
```

Out-of-order responses must work correctly.

When the server process dies, all pending requests must immediately unblock with `LspServerExited`.

No request may remain blocked indefinitely after process termination.

---

# 8. Server-to-client requests and notifications

Handle common operations including:

```text
window/logMessage
window/showMessage
window/workDoneProgress/create
$/progress

workspace/configuration
workspace/workspaceFolders
workspace/applyEdit

client/registerCapability
client/unregisterCapability

textDocument/publishDiagnostics
```

Unsupported notifications may be logged and ignored when safe.

Unsupported server requests must receive a valid JSON-RPC error response.

Never leave server requests unanswered indefinitely.

---

# 9. Initialization

Each client follows:

```text
spawn
  ↓
initialize
  ↓
InitializeResult
  ↓
capture server capabilities
  ↓
initialized
  ↓
configuration / dynamic registration
  ↓
ready
```

Populate appropriate initialization fields:

```text
processId
clientInfo
rootUri
workspaceFolders
capabilities
initializationOptions
```

Advertise only client capabilities that the subsystem actually implements.

---

# 10. Capability negotiation

Maintain normalized server capability state.

For example:

```python
@dataclass(frozen=True)
class LspCapabilities:
    diagnostics_pull: bool

    hover: bool
    definition: bool
    declaration: bool
    type_definition: bool
    implementation: bool
    references: bool

    document_symbols: bool
    workspace_symbols: bool

    completion: bool
    signature_help: bool

    code_actions: bool

    rename: bool
    prepare_rename: bool

    formatting: bool
    range_formatting: bool

    call_hierarchy: bool
    type_hierarchy: bool

    folding_ranges: bool
    selection_ranges: bool
    document_links: bool
    semantic_tokens: bool
    inlay_hints: bool
```

Capability state must derive from:

```text
initialize result
+
dynamic registrations
-
dynamic unregistrations
```

Support:

```text
client/registerCapability
client/unregisterCapability
```

Every semantic operation must check support before issuing its request.

Unsupported operations must raise:

```python
LspUnsupportedCapability
```

rather than blindly issuing the method.

---

# 11. Language detection

Centralize language handling.

At minimum:

```text
.ts    -> typescript
.tsx   -> typescriptreact

.js    -> javascript
.jsx   -> javascriptreact
.mjs   -> javascript
.cjs   -> javascript

.py    -> python

.java  -> java

.html  -> html
.htm   -> html

.css   -> css

.vue   -> vue
```

Provide:

```python
language_for_path(path)

is_supported_source_file(path)

server_for_language(language)
```

Do not scatter extension checks throughout client operations.

---

# 12. Project-root detection

Every source document must resolve to an appropriate server root.

Project-root detection must have a caller-provided maximum workspace boundary and must never walk above it.

### TypeScript / JavaScript / Vue

Search upward for:

```text
tsconfig.json
jsconfig.json
package.json
```

### Python

Search upward for:

```text
pyrightconfig.json
pyproject.toml
setup.py
setup.cfg
requirements.txt
Pipfile
```

### Java

Search upward for:

```text
pom.xml
build.gradle
build.gradle.kts
settings.gradle
settings.gradle.kts
.gradle/
```

### HTML / CSS

Prefer an associated frontend project root.

Otherwise use the supplied workspace root.

Project-root rules belong in centralized policy, not semantic operations.

---

# 13. URI and path primitives

Provide:

```python
path_to_uri(path: Path) -> str

uri_to_path(uri: str) -> Path
```

Support correctly:

```text
absolute paths
spaces
Unicode
Linux
macOS
Windows where practical
```

Do not manually construct `file://` URIs in individual methods.

---

# 14. Position primitives

Public positions are 1-based.

LSP positions are zero-based.

Define:

```python
@dataclass(frozen=True)
class SourcePosition:
    line: int
    column: int
```

and:

```python
@dataclass(frozen=True)
class SourceRange:
    start: SourcePosition
    end: SourcePosition
```

Provide:

```python
to_lsp_position(position)

from_lsp_position(position)

to_lsp_range(source_range)

from_lsp_range(lsp_range)
```

Individual semantic operations must not contain scattered `+1` / `-1` conversions.

---

# 15. Document synchronization

Each `LspClient` maintains document state:

```text
path
URI
languageId
version
last-known text or hash
open state
```

Provide:

```python
client.sync_document(path)
```

Required behavior:

```text
never opened
    -> textDocument/didOpen

already open + unchanged
    -> no protocol message

already open + changed
    -> increment version
    -> textDocument/didChange
```

Full-document synchronization is acceptable initially.

Semantic operations must synchronize the document before requesting information about it.

The subsystem must not depend solely on filesystem watcher behavior.

---

# 16. Diagnostics

Support both LSP diagnostic models:

```text
push:
    textDocument/publishDiagnostics

pull:
    textDocument/diagnostic
```

Use pull diagnostics when advertised.

Otherwise maintain push diagnostic state.

Normalize diagnostics into typed objects containing at least:

```text
URI/path
severity
message
source
code
range
related information
tags
document version/generation where known
```

Maintain:

```text
URI -> diagnostic state
```

---

# 17. Diagnostic freshness

Diagnostics returned after document changes must correspond to current synchronized contents.

This sequence:

```text
file changes
    ↓
sync_document
    ↓
diagnostics
```

must not return an older cached diagnostic state as authoritative.

For pull diagnostics:

```text
sync
→ request textDocument/diagnostic
```

For push diagnostics:

```text
sync
→ advance expected generation/version
→ wait for an appropriate publishDiagnostics update
```

Do not use arbitrary sleeps such as:

```python
time.sleep(2)
```

Use versions, generations, events, or equivalent synchronization.

Timeout must produce:

```python
LspDiagnosticsTimeout
```

not a false clean result.

---

# 18. Diagnostic severity

Normalize:

```text
error
warning
information
hint
```

Support threshold filtering:

```text
error
    errors only

warning
    errors + warnings

information
    errors + warnings + information

hint / all
    everything
```

---

# 19. Recursive diagnostics

Provide:

```python
manager.diagnostics_tree(
    root,
    *,
    level="error",
    max_files=...,
    max_diagnostics=...,
    timeout_per_file=...,
    excludes=...,
)
```

This is a first-class subsystem primitive.

Processing should be:

```text
discover supported files
        ↓
determine language
        ↓
determine project root
        ↓
group by
    (server identity, project root)
        ↓
obtain/reuse one client per group
        ↓
synchronize documents
        ↓
collect fresh diagnostics
        ↓
aggregate
```

Do not start a server per source file.

---

# 20. Recursive source discovery

Provide:

```python
discover_supported_files(
    workspace,
    root,
    excludes=(),
) -> Iterable[Path]
```

Requirements:

* recurse deterministically;
* include only supported languages;
* stay within the requested subtree;
* stay within the configured workspace boundary;
* do not follow symlink-directory loops;
* respect built-in exclusions;
* respect caller exclusions.

Built-in exclusions should include:

```text
.git
.hg
.svn

node_modules

dist
build
target
out
coverage

__pycache__

.venv
venv

.gradle
.idea
.cache
```

Exclusion policy must be centralized.

---

# 21. Recursive execution and partial failure

Independent clients may be processed concurrently.

Documents belonging to the same client should initially be processed sequentially unless safe concurrency is explicitly supported.

A failure in one server/project group must not automatically destroy results from other groups.

Represent recursive results with explicit completeness information.

For example:

```python
@dataclass(frozen=True)
class DiagnosticScanResult:
    diagnostics: tuple[LspDiagnostic, ...]
    files_checked: int
    files_discovered: int
    truncated_files: bool
    truncated_diagnostics: bool
    incomplete_groups: tuple[IncompleteGroup, ...]
```

Never confuse:

```text
no diagnostics
```

with:

```text
diagnostics unavailable
```

---

# 22. Semantic operations

The client should expose high-level operations.

## Hover

```python
client.hover(
    path,
    position,
) -> HoverResult | None
```

Uses:

```text
textDocument/hover
```

---

## Navigation

```python
client.go_to(
    path,
    position,
    kind,
)
```

Kinds:

```text
definition
declaration
type_definition
implementation
```

Map to:

```text
textDocument/definition
textDocument/declaration
textDocument/typeDefinition
textDocument/implementation
```

Capability-check before use.

---

## References

```python
client.references(
    path,
    position,
    include_declaration=False,
    limit=None,
)
```

Uses:

```text
textDocument/references
```

---

## Symbols

```python
client.document_symbols(path)

client.workspace_symbols(
    query,
    limit=None,
)
```

Use:

```text
textDocument/documentSymbol
workspace/symbol
```

Preserve document-symbol hierarchy when returned.

---

## Completion

```python
client.completion(
    path,
    position,
    trigger_character=None,
    limit=None,
)
```

Use:

```text
textDocument/completion
```

Normalize completion results instead of exposing raw protocol structures unnecessarily.

---

## Signature help

```python
client.signature_help(
    path,
    position,
    trigger_character=None,
)
```

Use:

```text
textDocument/signatureHelp
```

---

## Code actions

```python
client.code_actions(
    path,
    source_range,
    diagnostic_codes=None,
    kind=None,
)
```

Use:

```text
textDocument/codeAction
```

Listing actions must not automatically apply them.

---

## Rename

```python
client.rename(
    path,
    position,
    new_name,
) -> WorkspaceEdit
```

Use `prepareRename` when advertised.

Do not mutate files directly inside the protocol request.

Return a normalized `WorkspaceEdit`.

---

## Formatting

```python
client.format(path)

client.format_range(
    path,
    source_range,
)
```

Return edits.

Applying those edits is a separate operation.

---

## Hierarchy

Provide:

```python
client.call_hierarchy(
    path,
    position,
    direction,
)

client.type_hierarchy(
    path,
    position,
    direction,
)
```

Call directions:

```text
incoming
outgoing
```

Type directions:

```text
supertypes
subtypes
```

Use prepare requests where required.

---

# 23. Additional extensibility

The architecture should support without redesign:

```text
textDocument/documentHighlight
textDocument/foldingRange
textDocument/selectionRange
textDocument/documentLink
textDocument/semanticTokens/*
textDocument/codeLens
textDocument/inlayHint

workspace/executeCommand
```

These do not all need dedicated high-level methods initially.

---

# 24. WorkspaceEdit engine

Implement a shared WorkspaceEdit parser and application engine.

Support:

```text
changes
documentChanges

TextDocumentEdit

CreateFile
RenameFile
DeleteFile

versioned edits

multiple files
multiple edits per file
```

Provide separate primitives:

```python
normalize_workspace_edit(...)

preview_workspace_edit(...)

apply_workspace_edit(
    edit,
    *,
    workspace,
)
```

Safety requirements:

* validate all operations before mutation where practical;
* reject paths outside the permitted workspace;
* validate document versions when supplied;
* reject overlapping incompatible edits;
* apply same-file text edits against original contents;
* normally apply text edits in descending position order;
* do not silently partially apply multi-file edits;
* support Unicode and multiline edits correctly.

Server-initiated `workspace/applyEdit` must not silently mutate files unless explicitly enabled by the embedding application.

---

# 25. Progress and indexing

Handle:

```text
window/workDoneProgress/create
$/progress
```

Track progress state per client.

Expose enough state to distinguish:

```text
ready
indexing
busy
failed
dead
```

where the server provides sufficient information.

Diagnostics must not silently interpret an unstable/indexing server as clean when stable results cannot be established before timeout.

---

# 26. Server availability

Before spawning a configured server, resolve its executable.

Missing executables must produce:

```python
LspUnavailable
```

Cache unavailable state so repeated calls do not repeatedly attempt the same impossible startup.

Do not automatically install missing servers.

---

# 27. Crash recovery

If a server exits unexpectedly:

1. mark the client dead;
2. record exit code;
3. fail all pending requests;
4. preserve a bounded stderr tail;
5. expose the failure through status;
6. permit controlled restart;
7. apply restart backoff;
8. prevent crash loops.

One recursive scan must not restart the same crashing client for every file.

---

# 28. Stderr

Continuously drain server stderr.

Maintain a bounded ring buffer.

Expose useful failure context:

```text
Language server exited unexpectedly (code 1).

Last stderr:
...
```

Never retain or return unlimited stderr.

---

# 29. Errors

Create typed exceptions:

```text
LspError
├── LspUnavailable
├── LspUnsupportedCapability
├── LspStartupError
├── LspStartupTimeout
├── LspRequestError
├── LspRequestTimeout
├── LspProtocolError
├── LspServerExited
├── LspDocumentError
├── LspDiagnosticsTimeout
└── LspWorkspaceEditError
```

Do not collapse all failures into `RuntimeError`.

---

# 30. Configuration primitives

The package should accept typed configuration without assuming where configuration originates.

For example:

```python
@dataclass(frozen=True)
class LspConfig:
    enabled: bool = True
    startup_timeout: float = 30.0
    request_timeout: float = 15.0
    diagnostics_timeout: float = 10.0
```

Server configuration:

```python
@dataclass(frozen=True)
class ServerConfig:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    extensions: tuple[str, ...]
    initialization_options: Mapping[str, Any]
    settings: Mapping[str, Any]
```

Recursive configuration:

```python
@dataclass(frozen=True)
class RecursiveDiagnosticsConfig:
    max_files: int = 2000
    max_diagnostics: int = 250
    include_source: bool = False
    excludes: tuple[str, ...] = ()
```

The host application may construct these dataclasses from TOML, JSON, CLI options, or any other source.

The `lsp/` package should not care.

---

# 31. Language-specific requirements

## TypeScript / JavaScript

Use:

```text
typescript-language-server --stdio
```

Reuse a server for JavaScript and TypeScript sharing the same project root where appropriate.

Prefer workspace-local TypeScript where supported.

Do not install or modify project dependencies automatically.

---

## Python

Use:

```text
pyright-langserver --stdio
```

Respect existing project configuration and virtual environments.

Do not assume Pylance-only features exist.

Gate operations on advertised capabilities.

---

## Java

Use JDT LS.

Keep JDT LS startup/cache behavior isolated in its adapter.

Support Maven, Gradle, and plain Java where practical.

Store JDT LS metadata outside source repositories.

---

## HTML

Use:

```text
vscode-html-language-server --stdio
```

Expose advertised semantic capabilities.

---

## CSS

Use:

```text
vscode-css-language-server --stdio
```

Support `.css` initially.

Keep language mapping extensible for future `.scss` and `.less`.

---

## Vue

Use Vue Language Server.

Treat `.vue` as its own language.

Do not split Vue documents into independently managed HTML, CSS, and TypeScript documents.

Implement whatever TypeScript integration the Vue server requires while ensuring persistent services are reused.

---

# 32. Thread and process safety

Protect at least:

```text
request ID generation
pending request map
transport writes
client lifecycle
manager client map
document state
diagnostic state
dynamic capability state
recursive result aggregation
```

JSON-RPC writes must never interleave.

Avoid holding broad manager locks while blocking on server I/O.

---

# 33. Logging

Use normal structured Python logging.

Useful events include:

```text
[LSP pyright] starting root=...
[LSP pyright] initialized pid=...
[LSP pyright] -> textDocument/hover #14
[LSP pyright] <- #14 7ms
[LSP pyright] diagnostics uri=... count=3
[LSP recursive] files=184 diagnostics=20
[LSP pyright] exited code=1
```

Do not log full source documents.

Truncate large protocol payloads.

---

# 34. Security boundaries

The subsystem must not:

* download language servers automatically;
* evaluate server-supplied code;
* execute arbitrary shell strings received from a server;
* silently apply edits outside the permitted workspace;
* follow recursive symlink loops;
* allow recursive scans to escape their workspace boundary.

`workspace/executeCommand` is an LSP method.

It must not be interpreted as permission to execute an arbitrary local shell command.

---

# 35. Unit tests — transport

Use fake servers or in-memory streams.

Test:

```text
Content-Length framing
fragmented headers
fragmented bodies
multiple messages in one stream
out-of-order responses
concurrent requests
notifications
server requests
malformed messages
process death
request timeout
```

Tests must not depend exclusively on real language servers.

---

# 36. Unit tests — lifecycle and capabilities

Verify:

```text
initialize
initialized
```

ordering.

Verify initialization fields:

```text
clientInfo
rootUri
workspaceFolders
capabilities
initializationOptions
```

Test:

```text
dynamic registration
dynamic unregistration
unsupported capability rejection
clean shutdown
unexpected process death
restart behavior
```

---

# 37. Unit tests — document synchronization

Verify:

```text
first semantic request
    -> didOpen

second request unchanged
    -> no didChange

disk content changes
    -> didChange
    -> version incremented
```

Ensure the next semantic request sees current source contents.

---

# 38. Unit tests — diagnostics

Test:

```text
push diagnostics
pull diagnostics
severity normalization
threshold filtering
freshness after changes
diagnostic timeout
generation tracking
source ranges
result limits
```

Use synchronization primitives and fake servers.

Do not use long real sleeps.

---

# 39. Unit tests — recursive diagnostics

Test:

```text
supported files included
unsupported files ignored
nested traversal
built-in exclusions
caller exclusions
symlink loops
deterministic ordering

project-root grouping
server grouping
process reuse

severity thresholds
partial server failure
max_files
max_diagnostics
freshness between scans
```

Use mixed-language fixtures.

Verify multiple files belonging to one project/server reuse one process.

---

# 40. Unit tests — WorkspaceEdit

Cover:

```text
single edit
multiple same-file edits
multiple files
CreateFile
RenameFile
DeleteFile
version mismatch
Unicode
multiline edits
outside-workspace rejection
overlapping edits
atomic failure behavior
```

---

# 41. Integration fixtures

Create:

```text
tests/fixtures/lsp/
├── typescript/
├── javascript/
├── python/
├── java/
├── html/
├── css/
├── vue/
└── mixed/
```

Each language fixture should contain targets for whichever capabilities the server advertises:

```text
diagnostics
hover
definition
references
symbols
completion
signature help
rename
formatting
code actions
```

Skip real-server integration tests clearly when the executable is unavailable.

Unit tests must still cover the protocol independently.

---

# 42. Implementation phases

## Phase 1 — transport core

```text
LSP framing
JSON-RPC transport
reader/writer
request IDs
pending requests
notifications
server requests
stderr draining
process-death propagation
```

## Phase 2 — client lifecycle

```text
spawn
initialize
initialized
shutdown
capability normalization
dynamic registration
configuration requests
progress state
```

## Phase 3 — workspace primitives

```text
language detection
server selection
project-root detection
URI conversion
positions/ranges
document synchronization
```

## Phase 4 — diagnostics

```text
diagnostic normalization
push diagnostics
pull diagnostics
freshness
severity thresholds
timeouts
```

## Phase 5 — first real server

Prove one language server end-to-end with:

```text
status
diagnostics
hover
definition
references
server reuse
shutdown
```

Do not implement every server before the common core is proven.

## Phase 6 — recursive diagnostics

```text
file discovery
exclusions
grouping
client reuse
partial failures
bounded concurrency
aggregation
result limits
```

## Phase 7 — remaining read-only semantics

```text
symbols
completion
signature help
call hierarchy
type hierarchy
```

## Phase 8 — mutation support

```text
WorkspaceEdit
code actions
rename
formatting
```

## Phase 9 — remaining server adapters

```text
TypeScript / JavaScript
Python
HTML
CSS
Java
Vue
```

## Phase 10 — integration and regression

Run full unit and multi-language integration suites.

---

# 43. Required public surface

At completion, an embedding application should be able to use the subsystem roughly like this:

```python
manager = LspManager(
    workspace=workspace,
    config=config,
)

client = manager.get_client(
    workspace / "src/example.py"
)

diagnostics = client.diagnostics(
    workspace / "src/example.py"
)

hover = client.hover(
    workspace / "src/example.py",
    SourcePosition(
        line=18,
        column=12,
    ),
)

references = client.references(
    workspace / "src/example.py",
    SourcePosition(
        line=18,
        column=12,
    ),
)

project_diagnostics = manager.diagnostics_tree(
    workspace / "src",
    level="warning",
)

status = manager.status()

manager.shutdown_all()
```

For lower-level or future operations:

```python
result = client.request(
    "textDocument/documentHighlight",
    params,
)
```

The caller should never need to:

```text
frame Content-Length manually
manage JSON-RPC IDs
read server stdout
correlate responses
send didOpen/didChange manually
parse raw server capabilities
manage process reuse
track diagnostic generations
```

Those are responsibilities of the LSP subsystem.

---

# 44. Completion criteria

The subsystem is complete when:

* [ ] Correct LSP stdio framing works.
* [ ] Requests, notifications, responses, and server requests work.
* [ ] Concurrent and out-of-order request correlation works.
* [ ] Process death unblocks pending requests.
* [ ] Server stderr is continuously drained.
* [ ] Clean shutdown works.
* [ ] Persistent clients are reused by project root/server identity.
* [ ] Duplicate startup races are prevented.
* [ ] Controlled crash recovery works.
* [ ] Initialization follows the LSP lifecycle.
* [ ] Capability negotiation works.
* [ ] Dynamic registration and unregistration work.
* [ ] Unsupported operations fail explicitly.
* [ ] Language detection is centralized.
* [ ] Project-root detection works.
* [ ] Path/URI conversion works.
* [ ] Position/range conversion works.
* [ ] Document synchronization works.
* [ ] Push diagnostics work.
* [ ] Pull diagnostics work when advertised.
* [ ] Diagnostic freshness is reliable.
* [ ] Diagnostic severity thresholds work.
* [ ] Recursive source discovery works.
* [ ] Recursive exclusions work.
* [ ] Recursive diagnostics reuse clients.
* [ ] Recursive partial failures remain distinguishable from clean results.
* [ ] Recursive result limits are explicit.
* [ ] Hover works.
* [ ] Definition/declaration/type-definition/implementation are capability-gated.
* [ ] References work.
* [ ] Document/workspace symbols work.
* [ ] Completion works.
* [ ] Signature help works.
* [ ] Code actions work.
* [ ] WorkspaceEdit normalization/application works safely.
* [ ] Rename works where advertised.
* [ ] Formatting works where advertised.
* [ ] Call/type hierarchy works where advertised.
* [ ] TypeScript works.
* [ ] JavaScript works.
* [ ] Python works.
* [ ] Java works.
* [ ] HTML works.
* [ ] CSS works.
* [ ] Vue works.
* [ ] Missing language-server executables fail clearly.
* [ ] Unit tests exercise protocol behavior without real servers.
* [ ] Integration tests exercise all initial language adapters.
* [ ] Mixed-language recursive diagnostics work.

The core design rule is:

```text
Host application
    ↓
LspManager
    ↓
persistent LspClient
    ↓
LspTransport
    ↓
language-server process
```

High-level callers interact through normalized primitives:

```text
sync
diagnostics
hover
navigation
references
symbols
completion
signature help
code actions
rename
formatting
hierarchy
recursive diagnostics
status
shutdown
```

Raw JSON-RPC and server-process mechanics remain encapsulated inside the LSP subsystem.
