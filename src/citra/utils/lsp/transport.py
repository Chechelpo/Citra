"""Thread-safe JSON-RPC 2.0 transport using LSP stdio framing."""

from __future__ import annotations

from collections import deque
import json
import queue
import subprocess
from threading import Event, Lock, Thread
from typing import Any, Callable

from .errors import (
    LspProtocolError,
    LspRequestError,
    LspRequestTimeout,
    LspServerExited,
    LspTransportError,
)
from .protocol import make_error_response, make_notification, make_request, make_response


NotificationHandler = Callable[[str, Any], None]
RequestHandler = Callable[[str, Any], Any]


class JsonRpcTransport:
    """Own framing, correlation, reader threads, and server callbacks."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        notification_handler: NotificationHandler | None = None,
        request_handler: RequestHandler | None = None,
    ) -> None:
        """Initialize the instance."""
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise ValueError("Language-server process must have all stdio pipes.")
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._notification_handler = notification_handler
        self._request_handler = request_handler
        self._pending: dict[int, queue.Queue[dict[str, Any] | BaseException]] = {}
        self._pending_lock = Lock()
        self._write_lock = Lock()
        self._id_lock = Lock()
        self._next_id = 1
        self._closed = Event()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._reader = Thread(target=self._read_loop, name="citra-lsp-reader", daemon=True)
        self._stderr_reader = Thread(
            target=self._stderr_loop,
            name="citra-lsp-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()

    @property
    def process(self) -> subprocess.Popen[bytes]:
        """Handle process."""
        return self._process

    @property
    def stderr_tail(self) -> str:
        """Handle stderr tail."""
        return "".join(self._stderr_tail).strip()

    def request(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
        *,
        timeout: float,
    ) -> Any:
        """Handle request."""
        if timeout <= 0:
            raise ValueError("LSP request timeout must be positive.")
        with self._id_lock:
            request_id = self._next_id
            self._next_id += 1
        response_queue: queue.Queue[dict[str, Any] | BaseException] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue
        try:
            self._send(make_request(request_id, method, params))
            try:
                response = response_queue.get(timeout=timeout)
            except queue.Empty as error:
                if self._process.poll() is not None:
                    raise self._server_exited() from error
                raise LspRequestTimeout(
                    f"LSP request {method!r} timed out after {timeout:.1f}s."
                ) from error
            if isinstance(response, BaseException):
                raise response
            error_payload = response.get("error")
            if isinstance(error_payload, dict):
                raise LspRequestError(
                    int(error_payload.get("code", -32603)),
                    str(error_payload.get("message", "Language server request failed.")),
                    error_payload.get("data"),
                )
            return response.get("result")
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def notify(
        self,
        method: str,
        params: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Handle notify."""
        self._send(make_notification(method, params))

    def respond(self, request_id: int | str | None, result: Any) -> None:
        """Send a JSON-RPC success response to a server request."""
        self._send(make_response(request_id, result))

    def respond_error(
        self,
        request_id: int | str | None,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        """Send a JSON-RPC error response to a server request."""
        self._send(make_error_response(request_id, code, message, data))

    def close(self) -> None:
        """Handle close."""
        already_closed = self._closed.is_set()
        self._closed.set()
        try:
            self._stdin.close()
        except OSError:
            pass
        self._fail_pending(self._server_exited())
        if not already_closed:
            self._reader.join(timeout=0.5)
            self._stderr_reader.join(timeout=0.5)
        for stream in (self._stdout, self._stderr):
            try:
                stream.close()
            except OSError:
                pass

    def _send(self, message: dict[str, Any]) -> None:
        """Handle send."""
        if self._closed.is_set() or self._process.poll() is not None:
            raise self._server_exited()
        body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        framed = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        try:
            with self._write_lock:
                self._stdin.write(framed)
                self._stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise LspTransportError(f"Could not write to language server: {error}") from error

    def _read_loop(self) -> None:
        """Handle read loop."""
        try:
            while not self._closed.is_set():
                message = self._read_message()
                if message is None:
                    break
                self._dispatch(message)
        except BaseException as error:
            if not self._closed.is_set():
                self._fail_pending(error)
        finally:
            self._closed.set()
            self._fail_pending(self._server_exited())

    def _read_message(self) -> dict[str, Any] | None:
        """Handle read message."""
        headers: dict[str, str] = {}
        while True:
            line = self._stdout.readline()
            if line == b"":
                return None
            if line in {b"\r\n", b"\n"}:
                break
            try:
                name, value = line.decode("ascii").split(":", 1)
            except (UnicodeDecodeError, ValueError) as error:
                raise LspProtocolError(f"Malformed LSP header: {line!r}") from error
            headers[name.strip().casefold()] = value.strip()
        raw_length = headers.get("content-length")
        if raw_length is None:
            raise LspProtocolError("LSP message has no Content-Length header.")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise LspProtocolError(f"Invalid LSP Content-Length: {raw_length!r}") from error
        if not 0 <= length <= 64 * 1024 * 1024:
            raise LspProtocolError(f"Unsafe LSP message length: {length}")
        body = self._stdout.read(length)
        if len(body) != length:
            raise LspProtocolError("Language server closed mid-message.")
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LspProtocolError("Language server returned invalid JSON.") from error
        if not isinstance(message, dict):
            raise LspProtocolError("Language server returned a non-object JSON-RPC message.")
        return message

    def _dispatch(self, message: dict[str, Any]) -> None:
        """Handle dispatch."""
        request_id = message.get("id")
        method = message.get("method")
        if request_id is not None and method is None:
            with self._pending_lock:
                target = self._pending.get(request_id)
            if target is not None:
                target.put(message)
            return
        if isinstance(method, str) and request_id is not None:
            try:
                result = (
                    self._request_handler(method, message.get("params"))
                    if self._request_handler is not None
                    else None
                )
                response = make_response(request_id, result)
            except Exception as error:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(error)},
                }
            self._send(response)
            return
        if isinstance(method, str) and self._notification_handler is not None:
            self._notification_handler(method, message.get("params"))

    def _stderr_loop(self) -> None:
        """Handle stderr loop."""
        try:
            while True:
                line = self._stderr.readline()
                if not line:
                    return
                self._stderr_tail.append(line.decode("utf-8", errors="replace"))
        except OSError:
            return

    def _fail_pending(self, error: BaseException) -> None:
        """Handle fail pending."""
        with self._pending_lock:
            pending = tuple(self._pending.values())
        for target in pending:
            try:
                target.put_nowait(error)
            except queue.Full:
                pass

    def _server_exited(self) -> LspServerExited:
        """Handle server exited."""
        return LspServerExited(
            exit_code=self._process.poll(),
            stderr_tail=self.stderr_tail,
        )

