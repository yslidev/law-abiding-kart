#!/bin/zsh
# Run a simulator script. SuperTuxKart needs an awake display on macOS (it hangs at
# init if the screen is asleep), so wake the display and keep it on while running.
cd "$(dirname "$0")"
caffeinate -u -t 2 &
sleep 1
exec caffeinate -d -i .venv/bin/python -u "$@"
