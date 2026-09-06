#!/usr/bin/env sh
# Scan the public tree and optional commit identities without logging secrets.
exec "${PYTHON:-python3}" "$(dirname "$0")/check_public.py" "$@"
