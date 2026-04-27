#!/usr/bin/env bash
# Launch the Stretch Fruit Finder GUI (Level 4 / test_04_gui.py).
#
# This script is the single entry point used by the desktop icon, the
# applications grid entry, and any future shortcut. Resolves the package
# root from its own location so the launcher works no matter where the
# repo is cloned (e.g. ~/chef_ai/Chef-Stretch-Programs).
#
# Exit codes:
#   0 — GUI exited cleanly
#  >0 — Python process exited non-zero (full stack trace in the
#       attached terminal, since the .desktop file uses Terminal=true)

set -euo pipefail

# Resolve absolute paths regardless of how this script was invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# tkinter and pygame both need DISPLAY set. When a .desktop entry is
# launched from the GNOME shell, DISPLAY is normally inherited from the
# user session — but when launched from a fresh terminal-via-cron / SSH
# context we want a sensible fallback. :0 matches GDM's default.
export DISPLAY="${DISPLAY:-:0}"

# pygame's gamepad reader prefers SDL's "x11" video driver when DISPLAY
# is real, and silently falls back to "dummy" otherwise. Don't override.

cd "$PKG_ROOT"

# Prefer python3; the project has no venv pin, so use whatever python3
# the system / chef_ai install configured.
exec python3 bringup/test_04_gui.py "$@"
