#!/usr/bin/env python3
"""Validate local Markdown links inside the repository.

External URLs are intentionally skipped. The checker ignores fenced code blocks
and inline code, then verifies that relative Markdown links resolve to existing
files or directories.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


LINK_RE = re.compile(r"!\[[^\]\n]+\]\(([^)\n]+)\)|\[[^\]\n]+\]\(([^)\n]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}


def clean_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if ' "' in target:
        target = target.split(' "', 1)[0]
    if " '" in target:
        target = target.split(" '", 1)[0]
    return target


def iter_broken_links(root: Path):
    for path in root.rglob("*.md"):
        if ".git" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        in_fence = False
        for line_no, line in enumerate(lines, 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            scan = INLINE_CODE_RE.sub("", line)
            for match in LINK_RE.finditer(scan):
                target = clean_target(match.group(1) or match.group(2) or "")
                parsed = urlparse(target)
                if (
                    not target
                    or target.startswith("#")
                    or target.startswith("//")
                    or parsed.scheme in EXTERNAL_SCHEMES
                ):
                    continue

                no_fragment = target.split("#", 1)[0].split("?", 1)[0]
                if not no_fragment:
                    continue

                candidate = (
                    root / no_fragment.lstrip("/")
                    if no_fragment.startswith("/")
                    else path.parent / unquote(no_fragment)
                )
                if not candidate.exists():
                    yield {
                        "file": path.relative_to(root).as_posix(),
                        "line": line_no,
                        "target": target,
                        "resolved": candidate.relative_to(root).as_posix()
                        if str(candidate).lower().startswith(str(root).lower())
                        else str(candidate),
                    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    broken = list(iter_broken_links(root))
    if args.json:
        print(json.dumps({"broken_internal_links": len(broken), "links": broken}, indent=2))
    else:
        print(f"broken_internal_links={len(broken)}")
        for item in broken:
            print(f"{item['file']}:{item['line']}: {item['target']} -> {item['resolved']}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())

