#!/usr/bin/env python3
"""Fail if authoring-machine absolute paths leak into the repository.

Usage:
    python tools/check_paths.py
Exit code 0 = clean, 1 = absolute paths found.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"/mnt/|/home/|[A-Za-z]:\\\\|/Users/")
EXTS = {".py", ".csv", ".json", ".yaml", ".yml"}

hits = []
for path in ROOT.rglob("*"):
    if path == Path(__file__) or path.suffix.lower() not in EXTS or ".git" in path.parts:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for lineno, line in enumerate(text.splitlines(), 1):
        if PATTERN.search(line):
            hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:160]}")

if hits:
    print("Absolute paths found:")
    print("\n".join(hits))
    sys.exit(1)

print("OK: no absolute authoring paths in tracked text files.")
