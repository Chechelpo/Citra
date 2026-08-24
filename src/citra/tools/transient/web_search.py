from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import re
from typing import Any, override
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
    """Raised when the configured OpenSERP instance cannot complete a search."""


class WebSearch(Tool):
    INVALIDATES_TOOL_CACHE = False

    """
    Search the public web through the OpenSERP instance configured in the
    current ExecutionContext.

    A call may contain either one ``query`` or a batch of ``queries``. Batch
    queries share the same search options and are executed concurrently against
    OpenSERP's ``/mega/search`` endpoint. Successful and failed batch items are
    kept separate so one failed query does not discard the other results.

    Citra intentionally exposes a smaller API than OpenSERP itself. Provider-
    specific proxy, aggregation, and transport controls remain implementation
    details rather than model-facing search semantics.

    Result formats
    --------------

    text:
        Default. OpenSERP's compact plain-text representation optimized for
        model context.

    markdown:
        OpenSERP's rendered Markdown representation.

    json:
        Citra-normalized structured search results.

    ndjson:
        OpenSERP's newline-delimited JSON. For batch calls, each output line is
        wrapped with the originating query and batch index so provenance is not
        lost.

    Batch behavior
    --------------

    ``query`` and ``queries`` are mutually exclusive.

    Batch calls may contain up to ``MAX_BATCH_QUERIES`` non-empty queries.
    Duplicate queries are rejected because they would perform redundant network
    work.

    Batch searches run with bounded concurrency. A failed query is represented
    as an error for that query while successful queries are retained. If every
    query fails, the tool raises ``WebSearchError``.

    All search parameters other than ``query`` / ``queries`` are shared by the
    entire batch. ``timeout_seconds`` applies independently to each OpenSERP
    request rather than to the batch as a whole.
    """

    DEFAULT_ENGINES = (
        "bing",
        "duckduckgo",
        "google",
    )

    SUPPORTED_ENGINES = (
        "google",
        "bing",
        "duckduckgo",
        "ecosia",
        "yandex",
        "baidu",
    )

    SUPPORTED_FORMATS = (
        "text",
        "markdown",
        "json",
        "ndjson",
    )

    DEFAULT_MODE = "balanced"
    DEFAULT_FORMAT = "text"

    DEFAULT_MAX_RESULTS = 10
    MAX_RESULTS_LIMIT = 50

    DEFAULT_EXTRACT_RESULTS = 0
    MAX_EXTRACT_RESULTS = 5

    DEFAULT_TIMEOUT_SECONDS = 20.0
    MAX_TIMEOUT_SECONDS = 60.0

    MAX_BATCH_QUERIES = 10
    MAX_BATCH_WORKERS = 4

    # JSON extraction can contain an entire article or documentation page.
    # Bound each extracted result before returning structured data to the
    # model. Rendered text/Markdown responses are controlled by OpenSERP.
    MAX_EXTRACTED_CONTENT_LENGTH = 20_000

    DATE_RANGE_PATTERN = re.compile(
        r"^\d{8}\.\.\d{8}$"
    )

    DEFINITION = ChatCompletionTool(
        function=FunctionDefinition(
            name="web_search",
            description=(
                "Search the public web through OpenSERP. Supply either one "
                "'query' or a batch of 'queries'. Batch queries share the same "
                "search options and run concurrently. Use this for current or "
                "external research, documentation, specifications, upstream "
                "projects, issues, releases, and other information not "
                "authoritatively available in the local repository. Searches "
                "multiple engines by default. Returns compact plain text "
                "optimized for model consumption by default; Markdown, JSON, "
                "and NDJSON output are also available. Target-page extraction "
                "is optional."
            ),
            parameters=JsonSchema.object(
                properties=(
                    JsonProperty(
                        name="query",
                        schema=JsonSchema.string(
                            description=(
                                "One search query. Mutually exclusive with "
                                "'queries'. Be specific and include important "
                                "names, versions, errors, or constraints."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="queries",
                        schema=JsonSchema.array(
                            JsonSchema.string(),
                            description=(
                                "Batch of search queries. Mutually exclusive "
                                "with 'query'. All queries share the same search "
                                "options. Up to 10 queries may be supplied."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="format",
                        schema=JsonSchema.string(
                            description=(
                                "Result format. 'text' returns OpenSERP's "
                                "compact plain-text representation optimized "
                                "for LLM context; 'markdown' returns rendered "
                                "Markdown; 'json' returns Citra-normalized "
                                "structured results; 'ndjson' returns "
                                "newline-delimited JSON. Defaults to 'text'."
                            ),
                            enum=SUPPORTED_FORMATS,
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="engines",
                        schema=JsonSchema.array(
                            JsonSchema.string(
                                enum=SUPPORTED_ENGINES,
                            ),
                            description=(
                                "Search engines to use, in preferred order. "
                                "Defaults to bing, duckduckgo, and google."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="mode",
                        schema=JsonSchema.string(
                            description=(
                                "Search strategy. 'balanced' searches all "
                                "selected engines in parallel and merges their "
                                "results; 'any' tries engines in order until one "
                                "succeeds; 'fast' uses the currently fastest "
                                "available selected engine. Defaults to "
                                "'balanced'."
                            ),
                            enum=(
                                "balanced",
                                "any",
                                "fast",
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="language",
                        schema=JsonSchema.string(
                            description=(
                                "Optional search language hint, such as 'EN', "
                                "'ES', or 'DE'. Engine behavior may vary."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="region",
                        schema=JsonSchema.string(
                            description=(
                                "Optional market/location hint such as 'US', "
                                "'DE', 'en-GB', or a supported engine-specific "
                                "region."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="date_range",
                        schema=JsonSchema.string(
                            description=(
                                "Optional date interval in "
                                "YYYYMMDD..YYYYMMDD format."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="site",
                        schema=JsonSchema.string(
                            description=(
                                "Optional domain restriction, for example "
                                "'github.com' or 'docs.python.org'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="file_type",
                        schema=JsonSchema.string(
                            description=(
                                "Optional file-extension filter such as 'pdf', "
                                "'doc', or 'txt'."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="offset",
                        schema=JsonSchema.integer(
                            description=(
                                "Result pagination offset. Defaults to 0."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="max_results",
                        schema=JsonSchema.integer(
                            description=(
                                "Maximum number of search results returned per "
                                "query. Defaults to 10; Citra allows at most 50."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="extract_results",
                        schema=JsonSchema.integer(
                            description=(
                                "Fetch and embed cleaned page content for the "
                                "top N results of each query. 0 disables "
                                "extraction. Valid range is 0-5 and defaults "
                                "to 0."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="extract_mode",
                        schema=JsonSchema.string(
                            description=(
                                "Extraction strategy when extract_results > 0: "
                                "'auto' tries a fast fetch and renders when "
                                "needed, 'fast' avoids browser rendering, and "
                                "'rendered' forces browser rendering."
                            ),
                            enum=(
                                "auto",
                                "fast",
                                "rendered",
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="include_features",
                        schema=JsonSchema.boolean(
                            description=(
                                "Include supported SERP features such as answer "
                                "boxes, AI summaries, related searches, and "
                                "people-also-ask data. Defaults to false."
                            ),
                        ),
                        required=False,
                    ),
                    JsonProperty(
                        name="timeout_seconds",
                        schema=JsonSchema.number(
                            description=(
                                "Maximum seconds to wait for each OpenSERP "
                                "search request. Defaults to 20; maximum 60."
                            ),
                        ),
                        required=False,
                    ),
                ),
                additional_properties=False,
            ),
        )
    )

    def __init__(
        self,
        context: ExecutionContext,
    ) -> None:
        super().__init__(
            context=context,
            definition=self.DEFINITION,
        )

    @override
    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | str:
        queries, batch = self._queries(
            arguments
        )

        engines = self._engines(
            arguments
        )

        mode = str(
            arguments.get(
                "mode",
                self.DEFAULT_MODE,
            )
        ).lower()

        if mode not in {
            "balanced",
            "any",
            "fast",
        }:
            raise WebSearchError(
                f"Unsupported search mode: {mode}"
            )

        result_format = str(
            arguments.get(
                "format",
                self.DEFAULT_FORMAT,
            )
        ).lower()

        if result_format not in self.SUPPORTED_FORMATS:
            raise WebSearchError(
                f"Unsupported result format: {result_format}"
            )

        offset = arguments.get(
            "offset",
            0,
        )

        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
        ):
            raise WebSearchError(
                "'offset' must be a non-negative integer."
            )

        max_results = arguments.get(
            "max_results",
            self.DEFAULT_MAX_RESULTS,
        )

        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= self.MAX_RESULTS_LIMIT
        ):
            raise WebSearchError(
                "'max_results' must be between "
                f"1 and {self.MAX_RESULTS_LIMIT}."
            )

        extract_results = arguments.get(
            "extract_results",
            self.DEFAULT_EXTRACT_RESULTS,
        )

        if (
            not isinstance(extract_results, int)
            or isinstance(extract_results, bool)
            or not 0 <= extract_results <= self.MAX_EXTRACT_RESULTS
        ):
            raise WebSearchError(
                "'extract_results' must be between "
                f"0 and {self.MAX_EXTRACT_RESULTS}."
            )

        extract_mode = arguments.get(
            "extract_mode"
        )

        if (
            extract_mode is not None
            and extract_results == 0
        ):
            raise WebSearchError(
                "'extract_mode' requires 'extract_results' greater than 0."
            )

        timeout = arguments.get(
            "timeout_seconds",
            self.DEFAULT_TIMEOUT_SECONDS,
        )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= self.MAX_TIMEOUT_SECONDS
        ):
            raise WebSearchError(
                "'timeout_seconds' must be greater than 0 and "
                f"at most {self.MAX_TIMEOUT_SECONDS}."
            )

        date_range = arguments.get(
            "date_range"
        )

        if date_range is not None:
            date_range = str(
                date_range
            ).strip()

            if not self.DATE_RANGE_PATTERN.fullmatch(
                date_range
            ):
                raise WebSearchError(
                    "'date_range' must use YYYYMMDD..YYYYMMDD format."
                )

        base_params: dict[str, str | int] = {
            "engines": ",".join(
                engines
            ),
            "mode": mode,
            "limit": max_results,
            "start": offset,
            "dedupe": "true",
            "merge": "true",
            "features": (
                "true"
                if arguments.get(
                    "include_features",
                    False,
                )
                else "false"
            ),
            "extract": extract_results,
            "format": result_format,
        }

        language = self._optional_text(
            arguments,
            "language",
        )

        if language is not None:
            base_params["lang"] = language

        region = self._optional_text(
            arguments,
            "region",
        )

        if region is not None:
            base_params["region"] = region

        site = self._optional_text(
            arguments,
            "site",
        )

        if site is not None:
            base_params["site"] = site

        file_type = self._optional_text(
            arguments,
            "file_type",
        )

        if file_type is not None:
            base_params["file"] = file_type.lstrip(".")

        if date_range is not None:
            base_params["date"] = date_range

        if extract_mode is not None:
            base_params["extract_mode"] = str(
                extract_mode
            )

        search_arguments = {
            "base_params": base_params,
            "timeout": float(timeout),
            "result_format": result_format,
            "mode": mode,
            "engines": engines,
            "max_results": max_results,
            "include_features": arguments.get(
                "include_features",
                False,
            ),
        }

        if not batch:
            return self._search_one(
                queries[0],
                **search_arguments,
            )

        return self._search_batch(
            queries,
            **search_arguments,
        )

    @override
    def format_call_log(
        self,
        arguments: dict[str, Any],
    ) -> str:
        query = arguments.get(
            "query"
        )
        queries = arguments.get(
            "queries"
        )

        if query is not None:
            parts = [
                f"query={self._truncate(str(query), 120)}"
            ]

        elif isinstance(
            queries,
            list,
        ):
            parts = [
                f"queries={len(queries)}"
            ]

            if queries:
                parts.append(
                    "first="
                    + self._truncate(
                        str(queries[0]),
                        80,
                    )
                )

        else:
            parts = [
                "query=?"
            ]

        result_format = arguments.get(
            "format"
        )

        if (
            result_format is not None
            and result_format != self.DEFAULT_FORMAT
        ):
            parts.append(
                f"format={result_format}"
            )

        mode = arguments.get(
            "mode"
        )

        if mode is not None:
            parts.append(
                f"mode={mode}"
            )

        engines = arguments.get(
            "engines"
        )

        if engines:
            parts.append(
                "engines="
                + ",".join(
                    str(engine)
                    for engine in engines
                )
            )

        offset = arguments.get(
            "offset"
        )

        if offset:
            parts.append(
                f"offset={offset}"
            )

        max_results = arguments.get(
            "max_results"
        )

        if max_results is not None:
            parts.append(
                f"max={max_results}"
            )

        extract_results = arguments.get(
            "extract_results"
        )

        if extract_results:
            parts.append(
                f"extract={extract_results}"
            )

        site = arguments.get(
            "site"
        )

        if site:
            parts.append(
                f"site={site}"
            )

        return " | ".join(
            parts
        )

    @override
    def format_result_log(
        self,
        result: Any,
    ) -> str:
        if isinstance(
            result,
            str,
        ):
            if not result:
                return "empty output"

            return (
                f"{len(result.splitlines())} line(s) | "
                f"{len(result)} chars"
            )

        if not isinstance(
            result,
            dict,
        ):
            return str(
                result
            )

        if result.get(
            "batch"
        ) is True:
            total = result.get(
                "query_count",
                0,
            )
            succeeded = result.get(
                "successful_queries",
                0,
            )
            failed = result.get(
                "failed_queries",
                0,
            )

            parts = [
                f"{succeeded}/{total} query(s) succeeded"
            ]

            if failed:
                parts.append(
                    f"{failed} failed"
                )

            return " | ".join(
                parts
            )

        count = result.get(
            "returned_results",
            0,
        )

        responded = result.get(
            "engines_responded",
            [],
        )

        failed = result.get(
            "engines_failed",
            [],
        )

        parts = [
            f"{count} result(s)"
        ]

        if responded:
            parts.append(
                "engines="
                + ",".join(
                    str(engine)
                    for engine in responded
                )
            )

        if failed:
            parts.append(
                "failed="
                + ",".join(
                    str(engine)
                    for engine in failed
                )
            )

        took_ms = result.get(
            "took_ms"
        )

        if took_ms is not None:
            parts.append(
                f"{took_ms}ms"
            )

        return " | ".join(
            parts
        )

    def _search_one(
        self,
        query: str,
        *,
        base_params: dict[str, str | int],
        timeout: float,
        result_format: str,
        mode: str,
        engines: tuple[str, ...],
        max_results: int,
        include_features: bool,
    ) -> dict[str, Any] | str:
        params = {
            **base_params,
            "text": query,
        }

        payload = self._request(
            params=params,
            timeout=timeout,
            result_format=result_format,
        )

        if result_format != "json":
            if not isinstance(
                payload,
                str,
            ):
                raise WebSearchError(
                    "OpenSERP returned an unexpected rendered response type."
                )

            return payload

        if not isinstance(
            payload,
            dict,
        ):
            raise WebSearchError(
                "OpenSERP returned an unexpected JSON response type."
            )

        return self._normalize_json_response(
            payload=payload,
            query=query,
            mode=mode,
            engines=engines,
            max_results=max_results,
            include_features=include_features,
        )

    def _search_batch(
        self,
        queries: tuple[str, ...],
        *,
        base_params: dict[str, str | int],
        timeout: float,
        result_format: str,
        mode: str,
        engines: tuple[str, ...],
        max_results: int,
        include_features: bool,
    ) -> dict[str, Any] | str:
        results: list[dict[str, Any]] = [
            {
                "query": query,
            }
            for query in queries
        ]

        worker_count = min(
            len(queries),
            self.MAX_BATCH_WORKERS,
        )

        futures: dict[
            Future[dict[str, Any] | str],
            int,
        ] = {}

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="citra-web-search",
        ) as executor:
            for index, query in enumerate(
                queries
            ):
                future = executor.submit(
                    self._search_one,
                    query,
                    base_params=base_params,
                    timeout=timeout,
                    result_format=result_format,
                    mode=mode,
                    engines=engines,
                    max_results=max_results,
                    include_features=include_features,
                )

                futures[future] = index

            for future in as_completed(
                futures
            ):
                index = futures[
                    future
                ]

                try:
                    result = future.result()

                except WebSearchError as error:
                    results[index][
                        "error"
                    ] = str(
                        error
                    )

                else:
                    results[index][
                        "result"
                    ] = result

        successful = sum(
            1
            for item in results
            if "result" in item
        )

        failed = (
            len(results)
            - successful
        )

        if successful == 0:
            first_error = next(
                (
                    str(item["error"])
                    for item in results
                    if item.get("error")
                ),
                "unknown error",
            )

            raise WebSearchError(
                f"All {len(queries)} batch searches failed. "
                f"First error: {first_error}"
            )

        if result_format == "json":
            return {
                "batch": True,
                "query_count": len(
                    queries
                ),
                "successful_queries": successful,
                "failed_queries": failed,
                "results": results,
            }

        if result_format == "markdown":
            return self._format_markdown_batch(
                results
            )

        if result_format == "ndjson":
            return self._format_ndjson_batch(
                results
            )

        return self._format_text_batch(
            results
        )

    @staticmethod
    def _format_text_batch(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Combine OpenSERP plain-text responses while preserving query identity.

        Each individual response remains untouched apart from surrounding batch
        headers.
        """
        sections: list[str] = []

        for index, item in enumerate(
            results,
            start=1,
        ):
            query = WebSearch._display_query(
                str(item["query"])
            )

            header = (
                f"[Query {index}] {query}"
            )

            if "error" in item:
                body = (
                    "ERROR: "
                    + str(item["error"])
                )

            else:
                body = str(
                    item.get(
                        "result",
                        "",
                    )
                ).strip()

            sections.append(
                f"{header}\n{body}"
            )

        return "\n\n".join(
            sections
        )

    @staticmethod
    def _format_markdown_batch(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Combine OpenSERP Markdown responses under per-query headings.
        """
        sections: list[str] = []

        for index, item in enumerate(
            results,
            start=1,
        ):
            query = WebSearch._display_query(
                str(item["query"])
            )

            header = (
                f"## Query {index}: {query}"
            )

            if "error" in item:
                body = (
                    "**Error:** "
                    + str(item["error"])
                )

            else:
                body = str(
                    item.get(
                        "result",
                        "",
                    )
                ).strip()

            sections.append(
                f"{header}\n\n{body}"
            )

        return "\n\n".join(
            sections
        )

    @staticmethod
    def _format_ndjson_batch(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Preserve valid NDJSON while attaching batch provenance.

        Every OpenSERP result line is wrapped in:

        {
            "batch_index": ...,
            "query": ...,
            "result": ...
        }

        Failed queries produce one NDJSON error object.
        """
        lines: list[str] = []

        for index, item in enumerate(
            results,
            start=1,
        ):
            query = str(
                item["query"]
            )

            if "error" in item:
                lines.append(
                    json.dumps(
                        {
                            "batch_index": index,
                            "query": query,
                            "error": str(
                                item["error"]
                            ),
                        },
                        ensure_ascii=False,
                    )
                )

                continue

            raw = str(
                item.get(
                    "result",
                    "",
                )
            )

            for raw_line in raw.splitlines():
                raw_line = raw_line.strip()

                if not raw_line:
                    continue

                try:
                    result = json.loads(
                        raw_line
                    )

                except json.JSONDecodeError:
                    # Preserve unexpected provider output without breaking the
                    # overall NDJSON stream.
                    result = {
                        "raw": raw_line,
                    }

                lines.append(
                    json.dumps(
                        {
                            "batch_index": index,
                            "query": query,
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                )

        return "\n".join(
            lines
        )

    def _request(
        self,
        *,
        params: dict[str, str | int],
        timeout: float,
        result_format: str,
    ) -> dict[str, Any] | str:
        host = (
            self.context
            .web_search_config
            .host_url
            .rstrip("/")
        )

        url = (
            f"{host}/mega/search?"
            f"{urlencode(params)}"
        )

        accept = {
            "json": "application/json",
            "markdown": "text/markdown",
            "text": "text/plain",
            "ndjson": "application/x-ndjson",
        }[result_format]

        request = Request(
            url,
            headers={
                "Accept": accept,
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
            detail = self._http_error_detail(
                error
            )

            raise WebSearchError(
                f"OpenSERP returned HTTP {error.code}"
                + (
                    f": {detail}"
                    if detail
                    else ""
                )
            ) from error

        except URLError as error:
            raise WebSearchError(
                f"Could not connect to OpenSERP at {url}: "
                f"{error.reason}"
            ) from error

        except TimeoutError as error:
            raise WebSearchError(
                "OpenSERP search timed out."
            ) from error

        if result_format != "json":
            try:
                return raw.decode(
                    "utf-8"
                )

            except UnicodeDecodeError as error:
                raise WebSearchError(
                    "OpenSERP returned a non-UTF-8 rendered response."
                ) from error

        try:
            payload = json.loads(
                raw
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            raise WebSearchError(
                "OpenSERP returned an invalid JSON response."
            ) from error

        if not isinstance(
            payload,
            dict,
        ):
            raise WebSearchError(
                "OpenSERP returned an unexpected JSON response type."
            )

        return payload

    def _normalize_json_response(
        self,
        *,
        payload: dict[str, Any],
        query: str,
        mode: str,
        engines: tuple[str, ...],
        max_results: int,
        include_features: bool,
    ) -> dict[str, Any]:
        raw_results = payload.get(
            "results",
            [],
        )

        if not isinstance(
            raw_results,
            list,
        ):
            raise WebSearchError(
                "OpenSERP returned an invalid 'results' value."
            )

        results = [
            normalized
            for item in raw_results[:max_results]
            if isinstance(item, dict)
            if (
                normalized := self._normalize_result(
                    item
                )
            )
        ]

        meta = payload.get(
            "meta",
            {},
        )

        if not isinstance(
            meta,
            dict,
        ):
            meta = {}

        query_meta = payload.get(
            "query",
            {},
        )

        if not isinstance(
            query_meta,
            dict,
        ):
            query_meta = {}

        pagination = self._normalize_pagination(
            payload.get(
                "pagination"
            )
        )

        response: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "engines_requested": query_meta.get(
                "engines_requested",
                list(engines),
            ),
            "engines_responded": meta.get(
                "engines_responded",
                [],
            ),
            "engines_failed": meta.get(
                "engines_failed",
                [],
            ),
            "returned_results": len(
                results
            ),
            "results": results,
        }

        took_ms = meta.get(
            "took_ms"
        )

        if took_ms is not None:
            response["took_ms"] = took_ms

        engine_errors = meta.get(
            "engine_errors"
        )

        if engine_errors:
            response["engine_errors"] = engine_errors

        if pagination:
            response["pagination"] = pagination

        if include_features:
            raw_features = payload.get(
                "serp_features",
                [],
            )

            if isinstance(
                raw_features,
                list,
            ):
                features = [
                    normalized
                    for item in raw_features
                    if isinstance(item, dict)
                    if (
                        normalized := self._normalize_feature(
                            item
                        )
                    )
                ]

                if features:
                    response["serp_features"] = features

        return response

    @classmethod
    def _normalize_result(
        cls,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Reduce OpenSERP result objects to fields useful to the model.

        Provider/UI metadata such as favicons and low-level position details
        are intentionally omitted.
        """
        normalized: dict[str, Any] = {
            "rank": result.get(
                "rank"
            ),
            "type": result.get(
                "type"
            ),
            "title": result.get(
                "title"
            ),
            "url": result.get(
                "url"
            ),
            "snippet": result.get(
                "snippet"
            ),
            "domain": result.get(
                "domain"
            ),
            "engine": result.get(
                "engine"
            ),
        }

        classification = result.get(
            "classification"
        )

        if isinstance(
            classification,
            dict,
        ):
            useful_classification = {
                key: classification.get(
                    key
                )
                for key in (
                    "content_type",
                    "source_hint",
                )
                if classification.get(
                    key
                )
                not in (
                    None,
                    "",
                )
            }

            if useful_classification:
                normalized[
                    "classification"
                ] = useful_classification

        extracted = result.get(
            "extracted"
        )

        if isinstance(
            extracted,
            dict,
        ):
            extracted_content = extracted.get(
                "content"
            )

            normalized_extracted: dict[
                str,
                Any,
            ] = {
                "title": extracted.get(
                    "title"
                ),
                "format": extracted.get(
                    "format"
                ),
                "mode": extracted.get(
                    "mode_used"
                ),
            }

            if isinstance(
                extracted_content,
                str,
            ):
                normalized_extracted[
                    "content"
                ] = cls._truncate(
                    extracted_content,
                    cls.MAX_EXTRACTED_CONTENT_LENGTH,
                )

            normalized_extracted = {
                key: value
                for key, value
                in normalized_extracted.items()
                if value
                not in (
                    None,
                    "",
                    [],
                    {},
                )
            }

            if normalized_extracted:
                normalized[
                    "extracted"
                ] = normalized_extracted

        return {
            key: value
            for key, value
            in normalized.items()
            if value
            not in (
                None,
                "",
                [],
                {},
            )
        }

    @staticmethod
    def _normalize_feature(
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "type": feature.get(
                "type"
            ),
            "text": feature.get(
                "text"
            ),
            "engine": feature.get(
                "engine"
            ),
            "confidence": feature.get(
                "confidence"
            ),
        }

        links = feature.get(
            "links"
        )

        if isinstance(
            links,
            list,
        ):
            normalized_links = []

            for link in links[:10]:
                if not isinstance(
                    link,
                    dict,
                ):
                    continue

                item = {
                    "title": link.get(
                        "title"
                    ),
                    "url": link.get(
                        "url"
                    ),
                }

                item = {
                    key: value
                    for key, value
                    in item.items()
                    if value
                    not in (
                        None,
                        "",
                    )
                }

                if item:
                    normalized_links.append(
                        item
                    )

            if normalized_links:
                normalized["links"] = normalized_links

        return {
            key: value
            for key, value
            in normalized.items()
            if value
            not in (
                None,
                "",
                [],
                {},
            )
        }

    @staticmethod
    def _normalize_pagination(
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(
            value,
            dict,
        ):
            return {}

        normalized = {
            "page": value.get(
                "page"
            ),
            "has_more": value.get(
                "has_more"
            ),
            "next_offset": value.get(
                "next_start"
            ),
        }

        return {
            key: item
            for key, item
            in normalized.items()
            if item is not None
        }

    def _queries(
        self,
        arguments: dict[str, Any],
    ) -> tuple[tuple[str, ...], bool]:
        """
        Validate the mutually-exclusive single-query and batch-query inputs.

        Returns:
            (queries, batch)

        ``batch`` reflects which API form the model requested rather than the
        number of queries. A ``queries`` array containing one element therefore
        still receives batch-shaped output.
        """
        query_supplied = (
            "query" in arguments
            and arguments.get("query") is not None
        )

        queries_supplied = (
            "queries" in arguments
            and arguments.get("queries") is not None
        )

        if query_supplied == queries_supplied:
            raise WebSearchError(
                "Supply exactly one of 'query' or 'queries'."
            )

        if query_supplied:
            query = str(
                arguments["query"]
            ).strip()

            if not query:
                raise WebSearchError(
                    "Search query cannot be empty."
                )

            return (
                query,
            ), False

        raw_queries = arguments.get(
            "queries"
        )

        if (
            not isinstance(raw_queries, list)
            or not raw_queries
        ):
            raise WebSearchError(
                "'queries' must contain at least one search query."
            )

        if len(raw_queries) > self.MAX_BATCH_QUERIES:
            raise WebSearchError(
                "'queries' may contain at most "
                f"{self.MAX_BATCH_QUERIES} search queries."
            )

        queries: list[str] = []
        seen: set[str] = set()

        for index, raw_query in enumerate(
            raw_queries,
            start=1,
        ):
            if not isinstance(
                raw_query,
                str,
            ):
                raise WebSearchError(
                    f"Query {index} in 'queries' must be a string."
                )

            query = raw_query.strip()

            if not query:
                raise WebSearchError(
                    f"Query {index} in 'queries' cannot be empty."
                )

            if query in seen:
                raise WebSearchError(
                    f"Duplicate batch query: {query}"
                )

            seen.add(
                query
            )

            queries.append(
                query
            )

        return tuple(
            queries
        ), True

    def _engines(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, ...]:
        raw = arguments.get(
            "engines"
        )

        if raw is None:
            return self.DEFAULT_ENGINES

        if (
            not isinstance(raw, list)
            or not raw
        ):
            raise WebSearchError(
                "'engines' must contain at least one search engine."
            )

        engines: list[str] = []

        for engine_raw in raw:
            engine = str(
                engine_raw
            ).strip().lower()

            if engine not in self.SUPPORTED_ENGINES:
                raise WebSearchError(
                    f"Unsupported search engine: {engine}"
                )

            if engine not in engines:
                engines.append(
                    engine
                )

        return tuple(
            engines
        )

    @staticmethod
    def _optional_text(
        arguments: dict[str, Any],
        name: str,
    ) -> str | None:
        value = arguments.get(
            name
        )

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            raise WebSearchError(
                f"'{name}' cannot be empty when supplied."
            )

        return text

    @staticmethod
    def _http_error_detail(
        error: HTTPError,
    ) -> str | None:
        try:
            raw = error.read()

        except Exception:
            return (
                str(error.reason)
                if error.reason
                else None
            )

        if not raw:
            return (
                str(error.reason)
                if error.reason
                else None
            )

        try:
            payload = json.loads(
                raw
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            text = raw.decode(
                "utf-8",
                errors="replace",
            ).strip()

            return (
                WebSearch._truncate(
                    text,
                    500,
                )
                if text
                else None
            )

        if not isinstance(
            payload,
            dict,
        ):
            return WebSearch._truncate(
                str(payload),
                500,
            )

        reason = payload.get(
            "reason"
        )

        message = (
            payload.get("message")
            or payload.get("error")
        )

        if reason and message:
            return (
                f"{reason}: {message}"
            )

        if reason:
            return str(
                reason
            )

        if message:
            return str(
                message
            )

        return WebSearch._truncate(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            500,
        )

    @staticmethod
    def _display_query(
        query: str,
    ) -> str:
        """Collapse whitespace for compact batch headings."""
        return " ".join(
            query.split()
        )

    @staticmethod
    def _truncate(
        value: str,
        limit: int,
    ) -> str:
        if len(value) <= limit:
            return value

        omitted = (
            len(value)
            - limit
        )

        return (
            value[:limit]
            + "\n"
            + f"... <truncated {omitted} characters>"
        )

