# Evaluation Prompts

This directory covers prompts used to evaluate model outputs, prompts, agents, RAG systems, classifiers, and AI workflows.

## Scope

- Rubric prompts, pairwise comparison prompts, factuality checks, citation checks, safety checks, style checks, task-completion checks, and regression tests.
- Prompt-as-judge workflows must track judge model, rubric, examples, calibration, human agreement, and failure modes.

## Reference Links

- OpenAI Evals: https://github.com/openai/evals
- Promptfoo: https://www.promptfoo.dev/
- Ragas: https://docs.ragas.io/
- DeepEval: https://docs.confident-ai.com/
- Stanford HELM: https://crfm.stanford.edu/helm/
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness

## Routing Rules

- Put production eval harnesses in MLOps.
- Put prompt techniques in `../Universal_Techniques/`.
- Put benchmark datasets in the datasets benchmark folder.
