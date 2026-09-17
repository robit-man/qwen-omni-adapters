#!/usr/bin/env bash
# Open the Jetson monitor in its own terminal once the desktop is up.
#
# Run from an autostart entry, where the Exec line cannot carry shell syntax:
# .desktop files reserve ';' and quotes, so the command lives here instead.
set -u

# Let the session settle before taking a window; otherwise the terminal can
# race the shell and open behind everything or not at all.
sleep "${JTOP_DELAY:-8}"

command -v jtop >/dev/null 2>&1 || {
  printf 'jtop is not installed; nothing to show.\n' >&2
  exit 0
}

for terminal in gnome-terminal x-terminal-emulator xfce4-terminal konsole; do
  command -v "$terminal" >/dev/null 2>&1 || continue
  case $terminal in
    gnome-terminal)
      exec "$terminal" --title="jtop — Jetson monitor" --geometry=120x40 -- \
        bash -c 'jtop; printf "\njtop exited. Press enter to close.\n"; read -r _'
      ;;
    *)
      exec "$terminal" -e \
        bash -c 'jtop; printf "\njtop exited. Press enter to close.\n"; read -r _'
      ;;
  esac
done

printf 'No terminal emulator found; skipping the jtop window.\n' >&2
