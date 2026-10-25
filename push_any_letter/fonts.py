"""Resolve a local font without silently changing recorded collision geometry."""
import hashlib
import os
from pathlib import Path
from .geometry import default_font


def resolve_recording_font(metadata, override=None):
    explicit = override or os.environ.get('PUSH_LETTER_FONT')
    candidates = [explicit] if explicit else [metadata.get('font_path'), default_font()]
    expected = metadata.get('font_sha256')
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and (not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected):
            return str(path)
    raise ValueError('Matching recording font not found. Supply --font or PUSH_LETTER_FONT. '
                     f'Expected SHA-256: {expected}. Font files are not distributed with the dataset.')
