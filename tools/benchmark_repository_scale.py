#!/usr/bin/env python3
"""Benchmark AIForge scale against public GitHub repositories.

The script uses only the Python standard library. It counts local repository
files/links and compares them with GitHub REST API metadata for competitor
repositories. README link counts are a conservative default; full competitor
link extraction requires cloning or archive downloads and should be run as a
separate, heavier audit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_REPOS = [
    "FELIPEACASTRO/AIForge",
    "josephmisiti/awesome-machine-learning",
    "ChristosChristofidis/awesome-deep-learning",
    "owainlewis/awesome-artificial-intelligence",
    "Hannibal046/Awesome-LLM",
    "steven2358/awesome-generative-ai",
    "Shubhamsaboo/awesome-llm-apps",
    "aishwaryanr/awesome-generative-ai-guide",
    "ethicalml/awesome-production-machine-learning",
    "academic/awesome-datascience",
    "ml-tooling/best-of-ml-python",
    "zhimin-z/awesome-awesome-machine-learning",
    "ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code",
]

URL_RE = re.compile(r"https?://[^\s<>\"')]+")


def request_json(url: str, token: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "AIForge-scale-benchmark"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url: str, token: str | None = None) -> str:
    headers = {"User-Agent": "AIForge-scale-benchmark"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def local_metrics(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    dirs: list[Path] = []
    text_lines = 0
    text_words = 0
    url_mentions: list[str] = []

    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir():
            dirs.append(path)
            continue
        files.append(path)
        if path.suffix.lower() in {".md", ".txt", ".html", ".xml", ".yml", ".yaml", ".json", ".cff", ".webmanifest"} or path.name in {".gitignore", "LICENSE"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            text_lines += text.count("\n") + (1 if text else 0)
            text_words += len(re.findall(r"\w+", text))
            url_mentions.extend(url.rstrip(".,;:]") for url in URL_RE.findall(text))

    return {
        "root": str(root),
        "files": len(files),
        "directories": len(dirs),
        "markdown_files": sum(1 for path in files if path.suffix.lower() == ".md"),
        "text_lines_est": text_lines,
        "text_words_est": text_words,
        "external_url_mentions": len(url_mentions),
        "unique_external_urls": len(set(url_mentions)),
    }


def github_repo_metrics(repo: str, token: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"repo": repo}
    try:
        meta = request_json(f"https://api.github.com/repos/{repo}", token)
        branch = meta.get("default_branch") or "HEAD"
        row.update(
            {
                "stars": meta.get("stargazers_count"),
                "forks": meta.get("forks_count"),
                "size_kb": meta.get("size"),
                "default_branch": branch,
                "pushed_at": meta.get("pushed_at"),
                "description": meta.get("description"),
            }
        )

        tree = request_json(
            f"https://api.github.com/repos/{repo}/git/trees/{quote(branch)}?recursive=1",
            token,
        )
        blobs = [item for item in tree.get("tree", []) if item.get("type") == "blob"]
        row["tree_truncated"] = tree.get("truncated")
        row["files"] = len(blobs)
        row["markdown_files"] = sum(1 for item in blobs if item.get("path", "").lower().endswith(".md"))

        try:
            readme = request_text(f"https://raw.githubusercontent.com/{repo}/{branch}/README.md", token)
            urls = [url.rstrip(".,;:]") for url in URL_RE.findall(readme)]
            row["readme_bytes"] = len(readme.encode("utf-8"))
            row["readme_url_mentions"] = len(urls)
            row["readme_unique_urls"] = len(set(urls))
        except (HTTPError, URLError, TimeoutError) as exc:
            row["readme_error"] = f"{type(exc).__name__}: {exc}"
    except (HTTPError, URLError, TimeoutError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def render_markdown(local: dict[str, Any], repos: list[dict[str, Any]]) -> str:
    lines = [
        "# Repository Scale Benchmark",
        "",
        "## Local Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "files",
        "directories",
        "markdown_files",
        "text_lines_est",
        "text_words_est",
        "external_url_mentions",
        "unique_external_urls",
    ]:
        lines.append(f"| {key} | {local.get(key, 0)} |")

    lines.extend(
        [
            "",
            "## GitHub Comparator Snapshot",
            "",
            "| Repository | Stars | Files | Markdown files | README unique URLs | Pushed at |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in repos:
        lines.append(
            "| {repo} | {stars} | {files} | {markdown_files} | {readme_unique_urls} | {pushed_at} |".format(
                repo=row.get("repo", ""),
                stars=row.get("stars", ""),
                files=row.get("files", ""),
                markdown_files=row.get("markdown_files", ""),
                readme_unique_urls=row.get("readme_unique_urls", ""),
                pushed_at=row.get("pushed_at", ""),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Local repository root.")
    parser.add_argument("--repo", action="append", dest="repos", help="GitHub repo owner/name. Can be repeated.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between GitHub API calls.")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    root = Path(args.root).resolve()
    repos = args.repos or DEFAULT_REPOS

    local = local_metrics(root)
    remote_rows = []
    for repo in repos:
        remote_rows.append(github_repo_metrics(repo, token))
        time.sleep(args.delay)

    if args.format == "json":
        print(json.dumps({"local": local, "github": remote_rows}, indent=2))
    else:
        print(render_markdown(local, remote_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

