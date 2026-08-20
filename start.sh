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

CONFIG_PATH="$SCRIPT_DIR/.config/config.toml"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Citra virtual environment not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Citra configuration not found: $CONFIG_PATH" >&2
    exit 1
fi

# Permanent Citra configuration.
export CITRA_CONFIG_PATH="$CONFIG_PATH"

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
# disposable agent filesystem separately.
exec "$PYTHON" -m citra.main "$@"