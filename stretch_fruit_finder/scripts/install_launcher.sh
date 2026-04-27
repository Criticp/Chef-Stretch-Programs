#!/usr/bin/env bash
# Install a clickable "Chef Fruit Finder" launcher on the user's
# desktop and applications menu. Run this once per user, after each
# fresh git clone. Idempotent: safe to re-run after an update.
#
# What it does:
#   1. Marks scripts/run_gui.sh executable.
#   2. Renders chef-fruit-finder.desktop.in with the absolute path to
#      run_gui.sh and a chosen icon, writes the result to:
#        ~/.local/share/applications/chef-fruit-finder.desktop
#         (Activities / app grid)
#        ~/Desktop/chef-fruit-finder.desktop
#         (clickable icon on the GNOME desktop)
#   3. chmod +x both copies and marks them GIO-trusted so GNOME doesn't
#      show the "Untrusted launcher" warning.
#
# Run from anywhere:  scripts/install_launcher.sh
# (No args. Uses the script's own location to find the repo root.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER_PATH="$PKG_ROOT/scripts/run_gui.sh"
TEMPLATE_PATH="$PKG_ROOT/scripts/chef-fruit-finder.desktop.in"

# Use a stock GNOME icon name that's always present. Replace with a
# repo-bundled .png path later if we want to ship a custom icon.
ICON_PATH="applications-science"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "ERROR: template not found at $TEMPLATE_PATH" >&2
  exit 1
fi

if [[ ! -f "$LAUNCHER_PATH" ]]; then
  echo "ERROR: run_gui.sh not found at $LAUNCHER_PATH" >&2
  exit 1
fi

chmod +x "$LAUNCHER_PATH"

APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$APPS_DIR"
mkdir -p "$DESKTOP_DIR"

APPS_FILE="$APPS_DIR/chef-fruit-finder.desktop"
DESKTOP_FILE="$DESKTOP_DIR/chef-fruit-finder.desktop"

# Render the template — sed-replace the @PLACEHOLDER@ tokens.
render() {
  sed \
    -e "s|@LAUNCHER_PATH@|$LAUNCHER_PATH|g" \
    -e "s|@ICON_PATH@|$ICON_PATH|g" \
    "$TEMPLATE_PATH" > "$1"
  chmod +x "$1"
}

render "$APPS_FILE"
render "$DESKTOP_FILE"

# Mark trusted so GNOME launches it without the "Untrusted application
# launcher" prompt. `gio` is preferred on GNOME 42+; fall back silently.
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

# Refresh the desktop database so the Activities grid sees the new entry
# without a logout/login cycle. Optional — failures are non-fatal.
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi

echo "Installed launcher:"
echo "  $APPS_FILE"
echo "  $DESKTOP_FILE"
echo
echo "Look for 'Chef Fruit Finder' on the desktop and in the Activities grid."
echo "If the desktop icon shows as 'Untrusted', right-click -> Allow Launching."
