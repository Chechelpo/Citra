#!/usr/bin/env bash
set -euo pipefail

# Resolve start.sh to its real location, including when invoked through
# a symlink such as ~/.local/bin/citra.
SOURCE="${BASH_SOURCE[0]}"
SYMLINK_DEPTH=0
MAX_SYMLINK_DEPTH=40

while [[ -L "$SOURCE" ]]; do
    if (( SYMLINK_DEPTH >= MAX_SYMLINK_DEPTH )); then
        echo "Too many symbolic links while resolving Citra installation." >&2
        exit 1
    fi

    SOURCE_DIR="$(
        cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1
        pwd
    )"

    TARGET="$(readlink "$SOURCE")"

    if [[ "$TARGET" == /* ]]; then
        SOURCE="$TARGET"
    else
        SOURCE="$SOURCE_DIR/$TARGET"
    fi

    ((SYMLINK_DEPTH += 1))
done

SCRIPT_DIR="$(
    cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1
    pwd
)"

CITRA_ROOT="$SCRIPT_DIR/.citra"
CONFIG_DIR="$CITRA_ROOT/config"
LEGACY_CONFIG_PATH="$CITRA_ROOT/config.toml"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Citra virtual environment not found: $PYTHON" >&2
    exit 1
fi

if [[ -d "$CONFIG_DIR" ]]; then
    for config_file in tools.toml models.toml; do
        if [[ ! -f "$CONFIG_DIR/$config_file" ]]; then
            echo "Citra configuration not found: $CONFIG_DIR/$config_file" >&2
            exit 1
        fi
    done
    CONFIG_PATH="$CONFIG_DIR"
elif [[ -f "$LEGACY_CONFIG_PATH" ]]; then
    # Transitional compatibility for installations that have not yet split
    # their configuration. New installations should use .citra/config/*.toml.
    CONFIG_PATH="$LEGACY_CONFIG_PATH"
    echo "Warning: legacy Citra config detected at $LEGACY_CONFIG_PATH; migrate to $CONFIG_DIR/tools.toml and $CONFIG_DIR/models.toml (with optional $CONFIG_DIR/linting.toml)." >&2
else
    echo "Citra configuration directory not found: $CONFIG_DIR" >&2
    echo "Expected tools.toml and models.toml; linting.toml is optional." >&2
    exit 1
fi

# Permanent Citra configuration.
export CITRA_INSTALL_ROOT="$SCRIPT_DIR"
export CITRA_CONFIG_PATH="$CONFIG_PATH"
export CITRA_ROOT="$CITRA_ROOT"

# Import Citra from this installation only. Do not inherit the caller's
# PYTHONPATH, which could make unrelated Python packages shadow Citra.
export PYTHONPATH="$SCRIPT_DIR/src"

# Prevent host Python configuration from contaminating Citra's runtime.
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# Do not change directory here.
#
# The caller's current directory is intentionally preserved and becomes
# Citra's active workspace. WorkspaceContext creates and manages the
# lifecycle-scoped agent filesystem separately.
exec "$PYTHON" -m cProfile -o citra.prof -m citra.main "$@"