# Universal Prompting Techniques

This directory covers prompt patterns that apply across domains, models, and tasks.

## Scope

- Instruction clarity, role/context/task/output structure, examples, constraints, rubrics, and verification loops.
- Few-shot, zero-shot, decomposition, self-critique, tool-use prompting, retrieval-grounded prompting, structured outputs, and eval-driven prompt iteration.
- Prompt safety, injection resistance, uncertainty handling, citation requirements, and model-specific adaptation.

## Reference Links

- OpenAI prompt engineering: https://developers.openai.com/api/docs/guides/prompt-engineering
- Anthropic prompt engineering overview: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Google prompting strategies: https://ai.google.dev/gemini-api/docs/prompting-strategies
- OpenAI structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Promptfoo: https://www.promptfoo.dev/
- OpenAI Evals: https://github.com/openai/evals

## Routing Rules

- Put domain prompts in sibling prompt directories.
- Put prompt evaluation assets in `../Evaluation_Prompts/` or the MLOps evaluation section.
- Put agent prompts in the AI agents production section when tool use and state matter.
