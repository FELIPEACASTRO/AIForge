# Tools

> Maintenance and quality-assurance scripts for the AIForge index: benchmarking repository scale against competitors and validating internal Markdown links, so the resource catalog stays large, accurate, and link-clean.

## Contents

| Item | Description |
| --- | --- |
| [benchmark_repository_scale.py](benchmark_repository_scale.py) | Standard-library Python script that counts local files and links and compares AIForge scale with public GitHub repositories via the GitHub REST API. |
| [validate_internal_links.py](validate_internal_links.py) | Link checker that verifies relative Markdown links resolve to existing files or directories, ignoring external URLs and fenced/inline code. |

## Related

- Parent: [`../`](../)

**Keywords:** AIForge tools, repository benchmark, GitHub REST API, link validation, markdown link checker, internal links, quality assurance, Python scripts, repository scale, dead link detection, CI utilities, documentation maintenance
