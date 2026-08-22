"""LSP server-capability reflection.

The server returns a ``ServerCapabilities`` object during ``initialize``.
This module exposes a normalised, frozen :class:`LspCapabilities` view
plus boolean helpers so callers do not have to dig through the spec.

Capability state is derived from the ``initialize`` result and may be
updated through dynamic ``client/registerCapability`` /
``client/unregisterCapability`` notifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import LspUnsupportedCapability


@dataclass(frozen=True)
class LspCapabilities:
    """Normalised, immutable view of a server's declared capabilities."""

    # Pull diagnostics --------------------------------------------------------
    diagnostics_pull: bool = False

    # Text-document navigation -----------------------------------------------
    hover: bool = False
    definition: bool = False
    declaration: bool = False
    type_definition: bool = False
    implementation: bool = False
    references: bool = False

    # Symbols -----------------------------------------------------------------
    document_symbols: bool = False
    workspace_symbols: bool = False

    # Completion / signature --------------------------------------------------
    completion: bool = False
    signature_help: bool = False

    # Code actions / rename ---------------------------------------------------
    code_actions: bool = False
    rename: bool = False
    prepare_rename: bool = False

    # Formatting --------------------------------------------------------------
    formatting: bool = False
    range_formatting: bool = False

    # Hierarchies -------------------------------------------------------------
    call_hierarchy: bool = False
    type_hierarchy: bool = False

    # Structural features -----------------------------------------------------
    folding_ranges: bool = False
    selection_ranges: bool = False
    document_links: bool = False
    semantic_tokens: bool = False
    inlay_hints: bool = False

    # Raw server payload (kept for advanced inspection) ----------------------
    raw: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_server_capabilities(
        cls,
        caps: dict[str, Any] | None,
    ) -> LspCapabilities:
        """Build a :class:`LspCapabilities` from a raw ``ServerCapabilities`` dict."""
        caps = caps or {}

        def _has(key: str) -> bool:
            value = caps.get(key)
            # Provider options may legitimately be an empty object. Presence
            # still means the capability is supported unless the server
            # explicitly reports false/null.
            return value is not None and value is not False

        # ``textDocumentSync`` may be a bool, a number, or an object.
        # It governs synchronisation rather than a semantic capability, so
        # it is not modelled as a boolean field here.

        diag = caps.get("diagnosticProvider")
        diagnostics_pull = diag is not None and diag is not False

        rename_provider = caps.get("renameProvider")
        rename = rename_provider is not None and rename_provider is not False
        # ``renameProvider`` may be a bool or an object with ``prepareProvider``.
        prepare_rename = False
        if isinstance(rename_provider, dict):
            prepare_rename = bool(rename_provider.get("prepareProvider"))

        return cls(
            diagnostics_pull=diagnostics_pull,
            hover=_has("hoverProvider"),
            definition=_has("definitionProvider"),
            declaration=_has("declarationProvider"),
            type_definition=_has("typeDefinitionProvider"),
            implementation=_has("implementationProvider"),
            references=_has("referencesProvider"),
            document_symbols=_has("documentSymbolProvider"),
            workspace_symbols=_has("workspaceSymbolProvider"),
            completion=_has("completionProvider"),
            signature_help=_has("signatureHelpProvider"),
            code_actions=_has("codeActionProvider"),
            rename=rename,
            prepare_rename=prepare_rename,
            formatting=_has("documentFormattingProvider"),
            range_formatting=_has("documentRangeFormattingProvider"),
            call_hierarchy=_has("callHierarchyProvider"),
            type_hierarchy=_has("typeHierarchyProvider"),
            folding_ranges=_has("foldingRangeProvider"),
            selection_ranges=_has("selectionRangeProvider"),
            document_links=_has("documentLinkProvider"),
            semantic_tokens=_has("semanticTokensProvider"),
            inlay_hints=_has("inlayHintProvider"),
            raw=caps,
        )

    # ------------------------------------------------------------------
    # Capability checking
    # ------------------------------------------------------------------

    def require(self, capability: str) -> None:
        """Raise :class:`LspUnsupportedCapability` if *capability* is falsy.

        *capability* is the name of a boolean attribute on this dataclass.
        """
        if not getattr(self, capability, False):
            raise LspUnsupportedCapability(
                f"Server does not support capability '{capability}'."
            )

    def supports(self, capability: str) -> bool:
        """Return ``True`` when *capability* is truthy."""
        return bool(getattr(self, capability, False))

    # ------------------------------------------------------------------
    # Dynamic registration
    # ------------------------------------------------------------------

    def with_dynamic_registration(
        self,
        registrations: list[dict[str, Any]],
    ) -> LspCapabilities:
        """Return a new capabilities object reflecting dynamic registrations.

        Each registration dict is expected to follow the LSP
        ``Registration`` shape with ``id`` and ``method`` keys.  Supported
        methods are mapped to the corresponding boolean field.
        """
        return _apply_registrations(self, registrations, register=True)

    def with_dynamic_unregistration(
        self,
        unregistrations: list[dict[str, Any]],
    ) -> LspCapabilities:
        """Return a new capabilities object reflecting dynamic unregistrations."""
        return _apply_registrations(self, unregistrations, register=False)


# ---------------------------------------------------------------------------
# Mapping between LSP method names and capability fields
# ---------------------------------------------------------------------------

_METHOD_TO_FIELD: dict[str, str] = {
    "textDocument/hover": "hover",
    "textDocument/definition": "definition",
    "textDocument/declaration": "declaration",
    "textDocument/typeDefinition": "type_definition",
    "textDocument/implementation": "implementation",
    "textDocument/references": "references",
    "textDocument/documentSymbol": "document_symbols",
    "workspace/symbol": "workspace_symbols",
    "textDocument/completion": "completion",
    "textDocument/signatureHelp": "signature_help",
    "textDocument/codeAction": "code_actions",
    "textDocument/rename": "rename",
    "textDocument/prepareRename": "prepare_rename",
    "textDocument/formatting": "formatting",
    "textDocument/rangeFormatting": "range_formatting",
    "textDocument/prepareCallHierarchy": "call_hierarchy",
    "textDocument/prepareTypeHierarchy": "type_hierarchy",
    "textDocument/foldingRange": "folding_ranges",
    "textDocument/selectionRange": "selection_ranges",
    "textDocument/documentLink": "document_links",
    "textDocument/semanticTokens": "semantic_tokens",
    "textDocument/inlayHint": "inlay_hints",
    "textDocument/diagnostic": "diagnostics_pull",
}


def _apply_registrations(
    caps: LspCapabilities,
    items: list[dict[str, Any]],
    *,
    register: bool,
) -> LspCapabilities:
    """Apply a batch of registrations/unregistrations immutably."""
    changes: dict[str, bool] = {}
    for item in items:
        method = item.get("method")
        if not method:
            continue
        field = _METHOD_TO_FIELD.get(method)
        if field is None:
            continue
        changes[field] = register

    if not changes:
        return caps

    current = {f.name: getattr(caps, f.name) for f in caps.__dataclass_fields__.values()}
    current.update(changes)
    return LspCapabilities(**current)
