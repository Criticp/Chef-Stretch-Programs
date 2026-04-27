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

# When launched from a .desktop file, GNOME spawns a non-interactive
# shell that skips ~/.bashrc — so any Hello Robot env vars exported
# there (HELLO_FLEET_PATH / HELLO_FLEET_ID) are missing, and stretch_body
# fails to import with: "HELLO_FLEET_PATH". Re-derive them.
if [[ -z "${HELLO_FLEET_PATH:-}" ]]; then
  for rcfile in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile"; do
    if [[ -f "$rcfile" ]]; then
      # Pull just the export lines we care about. grep is safe even if
      # the lines sit past .bashrc's "[ -z $PS1 ] && return" guard,
      # which would defeat a plain `source` from a non-interactive shell.
      # shellcheck disable=SC2046
      eval "$(grep -E '^[[:space:]]*export[[:space:]]+HELLO_FLEET_(PATH|ID)=' "$rcfile" 2>/dev/null || true)"
    fi
    [[ -n "${HELLO_FLEET_PATH:-}" ]] && break
  done
fi

# Convention fallback: standard install puts the fleet at ~/stretch_user
# with a single stretch-re* directory inside.
if [[ -z "${HELLO_FLEET_PATH:-}" && -d "$HOME/stretch_user" ]]; then
  export HELLO_FLEET_PATH="$HOME/stretch_user"
fi
if [[ -z "${HELLO_FLEET_ID:-}" && -n "${HELLO_FLEET_PATH:-}" ]]; then
  fleet_id="$(ls "$HELLO_FLEET_PATH" 2>/dev/null | grep -m1 -E '^stretch-' || true)"
  if [[ -n "$fleet_id" ]]; then
    export HELLO_FLEET_ID="$fleet_id"
  fi
fi

if [[ -z "${HELLO_FLEET_PATH:-}" || -z "${HELLO_FLEET_ID:-}" ]]; then
  echo "ERROR: HELLO_FLEET_PATH and/or HELLO_FLEET_ID not set, and could not be auto-detected." >&2
  echo "Set them in ~/.bashrc (or ~/.profile), or run from a shell where they are exported." >&2
  exit 2
fi

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
