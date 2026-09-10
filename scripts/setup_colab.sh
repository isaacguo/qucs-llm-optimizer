#!/usr/bin/env bash
set -euo pipefail

QUCS_S_VERSION="${QUCS_S_VERSION:-26.1.1}"
QUCS_S_SHA256="${QUCS_S_SHA256:-a86ab951118bfdffc8acfda8893cde0e2aeedc7129cc73978ae7d16761ec5cb5}"
TOOLS_DIR="${QUCS_LLM_TOOLS_DIR:-$HOME/.local/share/qucs-llm-optimizer}"
BIN_DIR="$HOME/.local/bin"
APPIMAGE="$TOOLS_DIR/Qucs-S-${QUCS_S_VERSION}-linux-x86_64.AppImage"
APPDIR="$TOOLS_DIR/Qucs-S-${QUCS_S_VERSION}"
DOWNLOAD_URL="https://github.com/ra3xdh/qucs_s/releases/download/${QUCS_S_VERSION}/Qucs-S-${QUCS_S_VERSION}-linux-x86_64.AppImage"

if [[ ! -f pyproject.toml || ! -f training/grpo.py ]]; then
    echo "Run this script from the qucs-llm-optimizer repository root." >&2
    exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "The pinned Qucs-S AppImage requires an x86_64 Colab runtime." >&2
    exit 1
fi

mkdir -p "$TOOLS_DIR" "$BIN_DIR"

if [[ ! -f "$APPIMAGE" ]] || ! echo "$QUCS_S_SHA256  $APPIMAGE" | sha256sum --check --status; then
    echo "Downloading Qucs-S ${QUCS_S_VERSION}..."
    curl --fail --location --retry 3 --output "$APPIMAGE.tmp" "$DOWNLOAD_URL"
    echo "$QUCS_S_SHA256  $APPIMAGE.tmp" | sha256sum --check
    mv "$APPIMAGE.tmp" "$APPIMAGE"
    chmod +x "$APPIMAGE"
fi

if [[ ! -d "$APPDIR" ]]; then
    echo "Extracting Qucs-S AppImage without FUSE..."
    extract_dir="$(mktemp -d)"
    (
        cd "$extract_dir"
        "$APPIMAGE" --appimage-extract >/dev/null
    )
    mv "$extract_dir/squashfs-root" "$APPDIR"
    rmdir "$extract_dir"
fi

qucs_s_binary="$(find "$APPDIR" -type f -name qucs-s -print -quit)"
qucsator_binary="$(find "$APPDIR" -type f -name qucsator_rf -print -quit)"
if [[ -z "$qucs_s_binary" || -z "$qucsator_binary" ]]; then
    echo "The Qucs-S AppImage does not contain the expected binaries." >&2
    exit 1
fi

cat > "$BIN_DIR/qucs-s" <<EOF
#!/usr/bin/env bash
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="$APPDIR/usr/lib:$APPDIR/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
exec "$qucs_s_binary" "\$@"
EOF

cat > "$BIN_DIR/qucsator_rf" <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$APPDIR/usr/lib:$APPDIR/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
exec "$qucsator_binary" "\$@"
EOF
chmod +x "$BIN_DIR/qucs-s" "$BIN_DIR/qucsator_rf"

export PATH="$BIN_DIR:$PATH"
export QUCS_S="$BIN_DIR/qucs-s"
export QUCSATOR_RF="$BIN_DIR/qucsator_rf"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Installing the pinned Python 3.12 training environment..."
uv sync --extra train --frozen

cat > .env.colab <<EOF
PATH=$BIN_DIR:\$PATH
QUCS_S=$BIN_DIR/qucs-s
QUCSATOR_RF=$BIN_DIR/qucsator_rf
QT_QPA_PLATFORM=offscreen
EOF

echo "Running one real-Qucs reward as an end-to-end check..."
uv run qucs-grpo --dry-run --output-dir outputs/colab-dry-run

echo "Colab setup complete. Start training with:"
echo "  source .env.colab && uv run qucs-grpo --output-dir outputs/colab-grpo"