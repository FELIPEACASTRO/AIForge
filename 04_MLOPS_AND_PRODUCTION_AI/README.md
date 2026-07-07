# Deploy AI

Resources for deploying, managing, evaluating, securing, and monitoring AI models in production. This pillar connects research artifacts to reliable systems: reproducible pipelines, model serving, observability, cost controls, guardrails, CI/CD, and operational governance.

## Lifecycle Map

| Area | What belongs here |
|---|---|
| Cloud and platforms | AWS, Google Cloud, Azure, Databricks, Snowflake, IBM, model catalogs, managed inference, and AI app platforms. |
| MLOps and LLMOps | Experiment tracking, registries, prompt management, model governance, feature stores, evaluation workflows, and release controls. |
| Serving and inference | FastAPI, BentoML, TorchServe, Triton, vLLM, TensorRT-LLM, model gateways, routing, batching, streaming, and latency tuning. |
| Evaluation and monitoring | Offline evals, online evals, RAG evals, drift, hallucination checks, feedback loops, A/B tests, canaries, and benchmark harnesses. |
| Optimization | Quantization, pruning, distillation, sparsity, compilation, caching, speculative decoding, and hardware-aware serving. |
| Safety and guardrails | Red teaming, content filters, policy checks, PII controls, prompt injection defense, tool-use constraints, and audit logs. |
| Agents and integrations | Agent frameworks, tool calls, MCP, workflow orchestration, API connectors, state management, and human-in-the-loop controls. |

## Subcategories

- [Cloud Platforms](./Cloud_Platforms/) - AWS, Google Cloud, Azure, and managed AI platforms.
- [MLOps Platforms](./MLOps_Platforms/) - Tools for the full MLOps lifecycle, including W&B, MLflow, feature stores, monitoring, and registries.
- [Model Serving](./Model_Serving/) - Serving and deployment, including FastAPI, TorchServe, BentoML, and inference servers.
- [Model Optimization](./Model_Optimization/) - Compression, quantization, pruning, distillation, and efficient deployment.
- [Workflow Orchestration](./Workflow_Orchestration/) - Workflow orchestration and distributed systems, including Airflow, Prefect, Ray, and pipeline engines.
- [API Integration Tools](./API_Integration_Tools/) - Integration and utility tools around production AI systems.
- [AI_Agents](./AI_Agents/) - Agent architecture, tool use, state, orchestration, and production reliability patterns.
- [LLMOps_and_Prompt_Management](./LLMOps_and_Prompt_Management/) - Prompt versioning, prompt evals, context management, and LLM release operations.
- [Guardrails_and_Safety](./Guardrails_and_Safety/) - Policy enforcement, red teaming, safety checks, and secure model behavior.

## Cross-Cutting Source Maps

- [AI/ML data, model, prompt, agent, and benchmark source atlas](../03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/AI_ML_Data_Model_Prompt_Source_Atlas_Batch_01_2026-07-07.md) - official and high-authority sources for model catalogs, prompt engineering, agent frameworks, evaluation harnesses, red teaming, and benchmark suites.
