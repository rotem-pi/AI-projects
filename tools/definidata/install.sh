#!/usr/bin/env bash
# One-time setup: creates the venv, installs dependencies, and installs
# definiData as a real macOS app in /Applications (double-click, no
# terminal, no browser tab required after this).
#
#   ./install.sh
#
# For non-macOS or if you'd rather just run it in the browser, use init.sh
# instead.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "install.sh builds a macOS .app bundle and only works on macOS."
    echo "Use ./init.sh instead to run definiData in your browser."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "python3 is required. Install it and re-run this script."
    exit 1
fi

if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Checking AWS credentials..."
if ! python3 -c "import boto3; boto3.client('sts', region_name='eu-north-1').get_caller_identity()" &> /dev/null; then
    echo ""
    echo "Warning: no valid AWS credentials found right now. That's fine for"
    echo "install - just run 'aws login' (or 'aws sso login') before you open"
    echo "the app."
fi

# Skip Streamlit's interactive first-run "email address" prompt.
mkdir -p ~/.streamlit
if [ ! -f ~/.streamlit/credentials.toml ]; then
    printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
fi

echo "Building definiData.app..."
APP_NAME="definiData"
BUILD_DIR="$(mktemp -d)/${APP_NAME}.app"
mkdir -p "$BUILD_DIR/Contents/MacOS" "$BUILD_DIR/Contents/Resources"

cp "$PROJECT_DIR/assets/icon.icns" "$BUILD_DIR/Contents/Resources/icon.icns"

cat > "$BUILD_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>definiData</string>
    <key>CFBundleDisplayName</key>
    <string>definiData</string>
    <key>CFBundleIdentifier</key>
    <string>ai.definity.definidata</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>definiData</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
</dict>
</plist>
PLIST

cat > "$BUILD_DIR/Contents/MacOS/definiData" <<LAUNCHER
#!/bin/bash
exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/desktop_launcher.py"
LAUNCHER
chmod +x "$BUILD_DIR/Contents/MacOS/definiData"

DEST="/Applications/${APP_NAME}.app"
if [ -d "$DEST" ]; then
    echo "Removing previous install at $DEST..."
    rm -rf "$DEST"
fi

if cp -R "$BUILD_DIR" "$DEST" 2>/dev/null; then
    echo ""
    echo "Installed! Open definiData from your Applications folder or Launchpad."
else
    echo ""
    echo "Could not write to /Applications (permission denied)."
    echo "The app bundle is ready at: $BUILD_DIR"
    echo "Drag it into /Applications yourself, e.g.:"
    echo "  cp -R \"$BUILD_DIR\" /Applications/"
fi
