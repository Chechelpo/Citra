# src/citra/tools/web_search.py

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ...context import ExecutionContext
from ..tool import Tool
from ...utils.json_schema import (
    ChatCompletionTool,
    FunctionDefinition,
    JsonProperty,
    JsonSchema,
)


class WebSearchError(RuntimeError):
    """Raised when the configured SearXNG instance cannot complete a search."""


class WebSearch(Tool):
    """
    Searches the web using the SearXNG instance configured in the
    current ExecutionContext.

    The tool is transient and bound to a single ExecutionContext,
    like every other Citra tool.

    Model parameters
    ----------------
    query:
        Required search query.

    categories:
        Optional SearXNG search categories. Examples commonly include
        "general", "news", "images", "videos", "science", etc.
        Availability depends on the configured SearXNG instance.

    language:
        Search language code, for example "en", "es", "de", or "all".
        Defaults to "all".

    page:
        SearXNG result page to request. Starts at 1.

    time_range:
        Restrict results by publication/search time. Supported values:
        "day", "month", or "year".

    safe_search:
        Safe-search level:
            0 = disabled / normal
            1 = moderate
            2 = strict

    max_results:
        Maximum number of search results returned to the model.
        This is a Citra-side limit and is not sent to SearXNG.

    timeout_seconds:
        Maximum time Citra waits for the SearXNG HTTP request.
        This is a Citra-side option and is not sent to SearXNG.
    """

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="web_search",
            description=(
                "Search the public web using SearXNG. Use this when current, "
                "external, factual, or otherwise web-dependent information is "
                "required. Results contain titles, URLs, text excerpts, source "
                "engines, categories, scores, and publication dates when "
                "available."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="query",
                        schema=JsonSchema.string(
                            description=(
                                "The web search query. Make it specific and "
                                "include important names, terms, or constraints."
                            )
                        ),
                    ),
                    JsonProperty(
                        name="categories",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Optional SearXNG search categories to use, "
                                "such as 'general', 'news', 'science', "
                                "'images', or 'videos'. Available categories "
                                "depend on the configured SearXNG instance."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="language",
                        schema=JsonSchema.string(
                            description=(
                                "Language code for the search, such as 'en', "
                                "'es', 'de', or 'all'. Defaults to 'all'."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="page",
                        schema=JsonSchema.integer(
                            description=(
                                "Search result page number. Pages start at 1. "
                                "Defaults to 1."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="time_range",
                        schema=JsonSchema.string(
                            description=(
                                "Optional time restriction for engines that "
                                "support it."
                            ),
                            enum=(
                                "day",
                                "month",
                                "year",
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="safe_search",
                        schema=JsonSchema.integer(
                            description=(
                                "Safe-search level: 0 = normal/off, "
                                "1 = moderate, 2 = strict. Defaults to 0."
                            ),
                            enum=(0, 1, 2),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_results",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of results Citra should return "
                                "to the model. Defaults to 10."
                            )
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout_seconds",
                        schema=JsonSchema.number(
                            description=(
                                "Maximum number of seconds to wait for the "
                                "SearXNG server. Defaults to 15."
                            )
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        )
    )

    DEFAULT_MAX_RESULTS = 10
    MAX_RESULTS_LIMIT = 50

    DEFAULT_TIMEOUT_SECONDS = 15.0
    MAX_TIMEOUT_SECONDS = 60.0

    def __init__(self, context: ExecutionContext):
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        query: str = arguments["query"].strip()

        if not query:
            raise WebSearchError(
                "Search query cannot be empty."
            )

        page = arguments.get("page", 1)
        max_results = arguments.get(
            "max_results",
            self.DEFAULT_MAX_RESULTS,
        )
        timeout = arguments.get(
            "timeout_seconds",
            self.DEFAULT_TIMEOUT_SECONDS,
        )

        if page < 1:
            raise WebSearchError(
                "'page' must be greater than or equal to 1."
            )

        if not 1 <= max_results <= self.MAX_RESULTS_LIMIT:
            raise WebSearchError(
                "'max_results' must be between "
                f"1 and {self.MAX_RESULTS_LIMIT}."
            )

        if not 0 < timeout <= self.MAX_TIMEOUT_SECONDS:
            raise WebSearchError(
                "'timeout_seconds' must be greater than 0 and "
                f"at most {self.MAX_TIMEOUT_SECONDS}."
            )

        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "pageno": page,
            "safesearch": arguments.get("safe_search", 0),
        }

        language = arguments.get("language")

        if language:
            params["language"] = language

        categories = arguments.get("categories")

        if categories:
            params["categories"] = ",".join(categories)

        time_range = arguments.get("time_range")

        if time_range:
            params["time_range"] = time_range

        payload = self._request(
            params=params,
            timeout=timeout,
        )

        results = [
            self._normalize_result(result)
            for result in payload.get("results", [])[:max_results]
        ]

        return {
            "query": query,
            "page": page,
            "returned_results": len(results),
            "results": results,
            "answers": payload.get("answers", []),
            "suggestions": payload.get("suggestions", []),
            "corrections": payload.get("corrections", []),
        }

    def _request(
        self,
        params: dict[str, str | int],
        timeout: float,
    ) -> dict[str, Any]:
        host = self.context.web_search_config.host_url.rstrip("/")

        url = f"{host}/search?{urlencode(params)}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Citra/1.0",
            },
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                raw = response.read()

        except HTTPError as error:
            raise WebSearchError(
                "SearXNG returned HTTP "
                f"{error.code}: {error.reason}"
            ) from error

        except URLError as error:
            raise WebSearchError(
                f"Could not connect to SearXNG: {error.reason}"
            ) from error

        except TimeoutError as error:
            raise WebSearchError(
                "SearXNG search timed out."
            ) from error

        try:
            payload = json.loads(raw)

        except json.JSONDecodeError as error:
            raise WebSearchError(
                "SearXNG returned an invalid JSON response."
            ) from error

        if not isinstance(payload, dict):
            raise WebSearchError(
                "SearXNG returned an unexpected response type."
            )

        return payload

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Reduce the SearXNG result object to fields useful to the model.

        This prevents irrelevant SearXNG/UI metadata from unnecessarily
        consuming model context.
        """
        normalized = {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "published_date": result.get("publishedDate"),
            "category": result.get("category"),
            "score": result.get("score"),
            "engines": result.get("engines"),
        }

        return {
            key: value
            for key, value in normalized.items()
            if value not in (None, "", [], {})
        }