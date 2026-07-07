# AWS AI and Data Platforms

This directory covers AWS services, open-data registries, and cloud-native infrastructure used for AI, ML, data engineering, and inference.

## What Belongs Here

- platform docs, service capabilities, access patterns, pricing caveats, security controls, and quotas
- reference architectures, reproducible commands, environment variables, and deployment limitations
- fit notes for training, fine-tuning, batch inference, real-time serving, and governance

## Source Links

- [AWS machine learning](https://aws.amazon.com/machine-learning/)
- [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [AWS Registry of Open Data](https://registry.opendata.aws/)
- [Google Cloud AI](https://cloud.google.com/products/ai)
- [Azure AI services](https://azure.microsoft.com/en-us/products/ai-services)

## Evidence To Track

- Source URL, publication or update date, license, owner, and access conditions.
- Dataset/model/prompt version, benchmark split, metric, baseline, and known limitations when applicable.
- Clear separation between primary evidence, reproduced results, and AIForge interpretation.

## Routing Rules

- Put provider-neutral concepts in 04_MLOPS_AND_PRODUCTION_AI.
- Put datasets in dataset directories even when hosted by a cloud.
- Put model-specific notes in 02_LLM_AND_AI_MODELS.

## Next Enrichment Tasks

- Add high-authority papers, official docs, benchmark entries, and reproducible examples for this topic.
- Add local examples only when they include provenance, validation notes, and maintenance owner.
- Cross-link mature items to the relevant model, dataset, MLOps, or vertical-application directory.
