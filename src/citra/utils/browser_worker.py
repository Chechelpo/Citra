"""Sandbox-internal Playwright worker. Protocol: one JSON object per line."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse


MAX_TEXT = 100_000


def _origin(url: str) -> str:
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{scheme}://{parsed.hostname.lower()}{port}"


class BrowserWorker:
    def __init__(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError(
                "Playwright is not installed. Install Citra dependencies and "
                "run 'playwright install chromium'."
            ) from error

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-domain-reliability",
                "--disable-sync",
                "--metrics-recording-only",
                "--no-first-run",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            ],
        )
        self._context = self._browser.new_context(
            accept_downloads=True,
            service_workers="block",
        )
        self._page = self._context.new_page()
        self._allowed_origins: set[str] = set()
        self._refs: dict[str, Any] = {}
        self._console: list[str] = []
        self._errors: list[str] = []
        self._page.on(
            "console",
            lambda message: self._append(self._console, message.text),
        )
        self._page.on(
            "pageerror",
            lambda error: self._append(self._errors, str(error)),
        )
        self._page.route("**/*", self._route)
        if hasattr(self._page, "route_web_socket"):
            self._page.route_web_socket("**/*", self._route_web_socket)

    def dispatch(self, request: dict[str, Any]) -> Any:
        action = request["action"]
        if action == "open":
            url = str(request["url"])
            origin = _origin(url)
            if origin:
                self._allowed_origins.add(origin)
            response = self._page.goto(
                url,
                wait_until=request.get("wait_until", "domcontentloaded"),
                timeout=int(request.get("timeout_ms", 30_000)),
            )
            return {
                "url": self._page.url,
                "title": self._page.title(),
                "status": response.status if response is not None else None,
            }
        if action == "snapshot":
            return self._snapshot()
        if action == "click":
            self._locator(request).click(timeout=int(request.get("timeout_ms", 15_000)))
            return {"url": self._page.url}
        if action == "fill":
            self._locator(request).fill(
                str(request.get("value", "")),
                timeout=int(request.get("timeout_ms", 15_000)),
            )
            return {"url": self._page.url}
        if action == "press":
            self._locator(request).press(
                str(request["key"]),
                timeout=int(request.get("timeout_ms", 15_000)),
            )
            return {"url": self._page.url}
        if action == "wait_for":
            self._page.locator(str(request["selector"])).wait_for(
                state=str(request.get("state", "visible")),
                timeout=int(request.get("timeout_ms", 15_000)),
            )
            return {"url": self._page.url}
        if action == "screenshot":
            path = Path(str(request["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(path), full_page=bool(request.get("full_page", True)))
            return {"path": str(path), "url": self._page.url}
        if action == "console":
            return {"messages": self._console[-200:]}
        if action == "errors":
            return {"errors": self._errors[-200:]}
        if action == "reload":
            self._page.reload(wait_until="domcontentloaded")
            return {"url": self._page.url, "title": self._page.title()}
        if action == "back":
            self._page.go_back(wait_until="domcontentloaded")
            return {"url": self._page.url, "title": self._page.title()}
        if action == "forward":
            self._page.go_forward(wait_until="domcontentloaded")
            return {"url": self._page.url, "title": self._page.title()}
        raise ValueError(f"Unsupported browser action: {action}")

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._playwright.stop()

    def _route(self, route: Any) -> None:
        scheme = urlparse(route.request.url).scheme
        if scheme in {"about", "data", "blob"}:
            route.continue_()
            return
        origin = _origin(route.request.url)
        if origin and origin in self._allowed_origins:
            route.continue_()
        else:
            self._append(
                self._errors,
                f"Blocked unapproved URL: {route.request.url[:1000]}",
            )
            route.abort("blockedbyclient")

    def _route_web_socket(self, socket: Any) -> None:
        origin = _origin(socket.url)
        if origin and origin in self._allowed_origins:
            socket.connect_to_server()
        else:
            self._append(self._errors, f"Blocked unapproved WebSocket: {socket.url}")
            socket.close()

    def _locator(self, request: dict[str, Any]) -> Any:
        reference = request.get("ref")
        selector = request.get("selector")
        if reference is not None:
            locator = self._refs.get(str(reference))
            if locator is None:
                raise KeyError(f"Unknown or stale browser reference: {reference}")
            return locator
        if selector is not None:
            return self._page.locator(str(selector)).first
        raise ValueError("'ref' or 'selector' is required.")

    def _snapshot(self) -> dict[str, Any]:
        selectors = "a,button,input,textarea,select,[role=button],[tabindex]"
        locator = self._page.locator(selectors)
        count = min(locator.count(), 500)
        self._refs.clear()
        elements: list[dict[str, Any]] = []
        for index in range(count):
            item = locator.nth(index)
            reference = f"e{index + 1}"
            self._refs[reference] = item
            try:
                elements.append(
                    {
                        "ref": reference,
                        "tag": item.evaluate("element => element.tagName.toLowerCase()"),
                        "role": item.get_attribute("role"),
                        "text": (item.inner_text(timeout=1000) or "")[:500],
                        "aria_label": item.get_attribute("aria-label"),
                        "placeholder": item.get_attribute("placeholder"),
                    }
                )
            except Exception:
                continue
        body = self._page.locator("body").inner_text(timeout=5000)
        return {
            "url": self._page.url,
            "title": self._page.title(),
            "text": body[:MAX_TEXT],
            "elements": elements,
        }

    @staticmethod
    def _append(target: list[str], value: str) -> None:
        target.append(value[:4000])
        del target[:-500]


def main() -> None:
    worker: BrowserWorker | None = None
    try:
        worker = BrowserWorker()
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = worker.dispatch(request)
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response), flush=True)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
