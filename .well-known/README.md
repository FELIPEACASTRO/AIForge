# Well-Known Metadata

This directory covers machine-readable web metadata, discovery files, security contact files, and standards-based endpoints for deployed documentation.

## What Belongs Here

- standards-based machine-readable files used by browsers, crawlers, identity systems, or static-site tooling
- small policy pointers that explain how the file is served and validated
- checks that generated public paths match the repository and docs build

## Source Links

- [RFC 8615 well-known URIs](https://www.rfc-editor.org/rfc/rfc8615)
- [Security.txt](https://securitytxt.org/)
- [W3C DID Core](https://www.w3.org/TR/did-core/)
- [OpenSearch description format](https://github.com/dewitt/opensearch)
- [Web App Manifest](https://www.w3.org/TR/appmanifest/)

## Evidence To Track

- Source URL, publication or update date, license, owner, and access conditions.
- Dataset/model/prompt version, benchmark split, metric, baseline, and known limitations when applicable.
- Clear separation between primary evidence, reproduced results, and AIForge interpretation.

## Routing Rules

- Keep human narrative documentation in docs/.
- Do not store general AI resources or datasets in well-known metadata folders.
- Use this directory only for protocol-visible files and their local guide.

## Next Enrichment Tasks

- Add high-authority papers, official docs, benchmark entries, and reproducible examples for this topic.
- Add local examples only when they include provenance, validation notes, and maintenance owner.
- Cross-link mature items to the relevant model, dataset, MLOps, or vertical-application directory.
