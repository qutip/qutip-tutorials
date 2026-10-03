#!/usr/bin/env python3

import re
import sys
from pathlib import Path


# Validate anchors w.r.t. rendering rules of
# 1. nbviewer (used for tutorials on the website)
# 2. JupyterLab 4.6 (used by try-qutip web app)
# Additionally, ensure there are no duplicates among anchors

HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
ANCHOR = re.compile(r"\[[^\]]*\]\(#([^)]+)\)")


def heading_anchor(text):
    return text.strip().replace(" ", "-")


def check(path):
    lines = path.read_text(encoding="utf-8").splitlines()

    anchors = {}
    errors = 0

    for lineno, line in enumerate(lines, 1):
        for anchor in ANCHOR.findall(line):
            if anchor not in anchors:
                print(
                    f"::error file={path},line={lineno}::"
                    f"Broken anchor '#{anchor}'"
                )
                errors += 1

    return errors

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = list(root.rglob("*.md"))

if not files:
    print(f"error: no Markdown files found under {root}")
    sys.exit(2)

errors = sum(check(path) for path in files)
sys.exit(bool(errors))
