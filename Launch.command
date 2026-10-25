#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import push_any_letter' >/dev/null 2>&1; then
  .venv/bin/python -m pip install .
fi
exec .venv/bin/python -m push_any_letter.app "$@"
