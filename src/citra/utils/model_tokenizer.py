"""
Tokenizer implementation. 
Internally matches the model_id (trying to find an accurate family) to tokenizer functions that load tokenizers from `CITRA_INSTALL_ROOT/tokenizers` (Script).
Available tokenizers are:
❯ tree
.
├── claude.json
├── deepseekv4.json
├── gemma.model
├── glm.json
├── jamba.model
├── llama3.json
├── llama.model
├── mistral.model
├── nerdstash.model
├── nerdstash_v2.model
├── tiktoken.model
├── tokenizer_config.json
└── yi.model

Also caches tokenizer conversion in order to avoid retokenizing infinitely
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable


TOKENIZER_FILES = {
    "claude": "claude.json",
    "deepseekv4": "deepseekv4.json",
    "gemma": "gemma.model",
    "glm": "glm.json",
    "jamba": "jamba.model",
    "llama3": "llama3.json",
    "llama": "llama.model",
    "mistral": "mistral.model",
    "nerdstash": "nerdstash.model",
    "nerdstash_v2": "nerdstash_v2.model",
    "tiktoken": "tiktoken.model",
    "yi": "yi.model",
    "nemotron3": "nemotron3.json"
}


def _tokenizer_root() -> Path:
    """Handle tokenizer root."""
    root = os.environ.get("CITRA_INSTALL_ROOT")
    install_root = (
        Path(root).expanduser()
        if root
        else Path(__file__).resolve().parents[3]
    )
    path = install_root / "tokenizers"
    if not path.is_dir():
        raise RuntimeError(f"Tokenizer directory does not exist: {path}")

    return path


@lru_cache(maxsize=256)
def _model_family(model_id: str) -> str:
    """
    Best-effort mapping of a model id to one of our tokenizer families.
    """
    model = model_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", model).strip("-")

    # Order matters: more-specific families must come first.

    if "claude" in normalized or "anthropic" in normalized:
        return "claude"

    if "deepseek" in normalized:
        return "deepseekv4"

    if "gemma" in normalized:
        return "gemma"

    if any(x in normalized for x in ("chatglm", "glm-4", "glm4", "glm-3", "glm3")):
        return "glm"

    if "jamba" in normalized:
        return "jamba"

    if any(
        x in normalized
        for x in (
            "nerdstash-v2",
            "nerdstash-v-2",
            "nerdstash2",
            "nerdstash-2",
        )
    ):
        return "nerdstash_v2"

    if "nerdstash" in normalized:
        return "nerdstash"

    if any(
        x in normalized
        for x in (
            "llama-3",
            "llama3",
            "llama-4",
            "llama4",
            "meta-llama-3",
            "meta-llama-4",
        )
    ):
        return "llama3"

    # CodeLlama and Llama 1/2 use the older SentencePiece tokenizer.
    if "llama" in normalized:
        return "llama"

    if any(x in normalized for x in ("mistral", "mixtral", "ministral")):
        return "mistral"

    if (
        normalized == "yi"
        or normalized.startswith("yi-")
        or "/yi-" in model
        or "01-ai" in normalized
    ):
        return "yi"

    # OpenAI-family models.
    if any(
        x in normalized
        for x in (
            "gpt-",
            "chatgpt",
            "text-davinci",
            "text-embedding",
            "o1",
            "o3",
            "o4",
        )
    ):
        return "tiktoken"

    if (
        "nemotron-3" in normalized
        or "nemotron3" in normalized
    ):
        return "nemotron3"

    # Older Llama-based Nemotron models
    if "nemotron" in normalized and "llama" in normalized:
        return "llama3"
    
    # Most newer generic BPE models are closer to Llama 3 than the old
    # Llama SentencePiece vocabulary. This is intentionally only a fallback.
    return "llama3"


def _load_json_tokenizer(path: Path) -> Callable[[str], int]:
    """Handle load json tokenizer."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Loading JSON tokenizers requires the `tokenizers` package"
        ) from exc

    tokenizer = Tokenizer.from_file(str(path))

    def count(text: str) -> int:
        # We are measuring the supplied text, not constructing a model prompt,
        # so don't inject BOS/EOS or other post-processor tokens.
        """Handle count."""
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def _load_sentencepiece_tokenizer(path: Path) -> Callable[[str], int]:
    """Handle load sentencepiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Loading SentencePiece tokenizers requires the "
            "`sentencepiece` package"
        ) from exc

    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.Load(str(path)):
        raise RuntimeError(f"Could not load SentencePiece tokenizer: {path}")

    def count(text: str) -> int:
        """Handle count."""
        return len(tokenizer.EncodeAsIds(text))

    return count


def _load_tiktoken_tokenizer(path: Path) -> Callable[[str], int]:
    """
    Load a tiktoken BPE-ranks file.

    The local tiktoken.model is expected to use the normal tiktoken
    `<base64 token> <rank>` format.
    """
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        raise RuntimeError(
            "Loading the tiktoken tokenizer requires the `tiktoken` package"
        ) from exc

    mergeable_ranks = load_tiktoken_bpe(str(path))

    # The splitting regex isn't encoded in a .tiktoken-style rank file.
    # Reuse cl100k's regex; Citra's local rank table still determines the
    # actual vocabulary/merges.
    base = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.Encoding(
        name="citra-local",
        pat_str=base._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )

    def count(text: str) -> int:
        """Handle count."""
        return len(tokenizer.encode_ordinary(text))

    return count


@lru_cache(maxsize=len(TOKENIZER_FILES))
def _get_tokenizer(family: str) -> Callable[[str], int]:
    """
    Load/convert each tokenizer once.

    This is the important cache: loading/converting tokenizer files for every
    call can be considerably more expensive than encoding the text itself.
    """
    try:
        filename = TOKENIZER_FILES[family]
    except KeyError:
        raise ValueError(f"Unknown tokenizer family: {family!r}") from None

    path = _tokenizer_root() / filename
    if not path.is_file():
        raise RuntimeError(f"Tokenizer file does not exist: {path}")

    if path.suffix == ".json":
        return _load_json_tokenizer(path)

    if family == "tiktoken":
        # Some distributions name this `tiktoken.model` despite it not being
        # a SentencePiece model.
        return _load_tiktoken_tokenizer(path)

    return _load_sentencepiece_tokenizer(path)


@lru_cache(maxsize=1024)
def _cached_tokenize(family: str, text: str) -> int:
    """
    Small bounded cache for identical inputs.

    Bounded deliberately: caching arbitrary prompt strings forever would leak
    memory in a long-running server.
    """
    return _get_tokenizer(family)(text)


def tokenize(model_id: str, text: str) -> int:
    """
    Get the amount of tokens this text represents.

    The tokenizer is selected by model family and loaded lazily from
    $CITRA_INSTALL_ROOT/tokenizers.
    """
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    family = _model_family(model_id)
    return _cached_tokenize(family, text)
